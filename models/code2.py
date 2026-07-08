import torch.nn as nn

# ogbg-code2 model components: the AST node encoder and the sequence-prediction head wrapper.
# Both are code2-specific, so they live together here (mirroring the data-side utils/code2.py)
# rather than polluting the shared models/encoder.py.


class ASTNodeEncoder(nn.Module):
    '''
        1-argument node encoder for the ogbg-code2 AST.

        Unlike the OGB reference encoder (which takes ``forward(x, depth)``), the node depth is
        baked into ``x`` as column 2 by the ``AddDepthToX`` transform, so the encoder matches the
        single-argument ``node_encoder(x)`` interface every backbone in this repo expects.

        Input:
            x: (N, 3) long tensor = [node type idx, node attribute idx, AST depth].
        Output:
            (N, emb_dim) node embedding = type + attribute + (clamped) depth embeddings.
    '''

    def __init__(self, emb_dim, num_nodetypes, num_nodeattributes, max_depth=20):
        super().__init__()
        self.max_depth = max_depth
        self.type_encoder = nn.Embedding(num_nodetypes, emb_dim)
        self.attribute_encoder = nn.Embedding(num_nodeattributes, emb_dim)
        self.depth_encoder = nn.Embedding(max_depth + 1, emb_dim)

    def reset_parameters(self):
        for e in (self.type_encoder, self.attribute_encoder, self.depth_encoder):
            nn.init.xavier_uniform_(e.weight.data)

    def forward(self, x):
        depth = x[:, 2].clone()
        depth[depth > self.max_depth] = self.max_depth
        return self.type_encoder(x[:, 0]) + self.attribute_encoder(x[:, 1]) + self.depth_encoder(depth)


class Code2Model(nn.Module):
    '''
        Sequence-prediction wrapper for the ogbg-code2 task.

        Any backbone in this repo returns a pooled graph embedding when built with
        ``num_pred_heads=None`` (see models/gin.py, models/ldna_net.py). Code2Model turns that
        single embedding into ``max_seq_len`` independent per-position classifiers over the
        method-name vocabulary, so the backbones stay unchanged and dataset-specific logic lives
        here. It also carries ``idx2vocab`` so the training loop can detect the code2 task
        (``hasattr(model, 'idx2vocab')``) and decode predictions for the F1 evaluator.
    '''

    def __init__(self, backbone, emb_dim, max_seq_len, num_vocab, idx2vocab):
        super().__init__()
        self.backbone = backbone
        self.heads = nn.ModuleList([nn.Linear(emb_dim, num_vocab) for _ in range(max_seq_len)])
        self.max_seq_len = max_seq_len
        self.idx2vocab = idx2vocab

    def reset_parameters(self):
        self.backbone.reset_parameters()
        for h in self.heads:
            h.reset_parameters()

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.backbone(x, edge_index, edge_attr, batch)   # (num_graphs, emb_dim)
        return [head(h) for head in self.heads]              # list of max_seq_len (num_graphs, num_vocab)
