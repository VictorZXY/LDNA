import torch.nn as nn


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
