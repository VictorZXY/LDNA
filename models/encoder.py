from torch import nn


class Encoder(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, num_features):
        super(Encoder, self).__init__()
        
        self.embedding_list = nn.ModuleList()

        for i in range(num_features):
            self.embedding_list.append(nn.Embedding(num_embeddings, embedding_dim))

        self.reset_parameters()

    def reset_parameters(self):
        # Xavier rather than nn.Embedding's own N(0, 1): this encoder feeds the conv stack directly,
        # and the default is an order of magnitude wider at these widths (std 0.98 vs 0.084 for the
        # ZINC-sized Encoder(28, 256)). Defining the method also makes the per-run encoder reset in
        # `train.py` reproduce the construction-time distribution instead of falling through to the
        # nn.Embedding default and silently rescaling the model's input.
        for emb in self.embedding_list:
            nn.init.xavier_uniform_(emb.weight.data)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.embedding_list[i](x[:,i])

        return x_embedding
    