import torch.nn as nn


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
