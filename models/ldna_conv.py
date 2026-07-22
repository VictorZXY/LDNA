import torch
from torch import nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.models import MLP
from torch_scatter import scatter


class LDNAConv(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim=None, num_pre_layers=1, num_post_layers=1,
                 gate_hidden_channels=32, **kwargs):
        super(LDNAConv, self).__init__(aggr=None, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim

        if edge_dim is not None:
            self.pre_nn = MLP(in_channels=2 * in_channels + edge_dim, hidden_channels=in_channels,
                              out_channels=in_channels, num_layers=num_pre_layers)
        else:
            self.pre_nn = MLP(in_channels=2 * in_channels, hidden_channels=in_channels,
                              out_channels=in_channels, num_layers=num_pre_layers)

        # Learnable per-edge gate on the canonical-rank gap. Its input is the 3 rank-gap features
        # [delta, |delta|, sign(delta)] built in `message()` (hence `gate_in_channels = 3`); it maps
        # them, through a small bottleneck, to a per-channel weight so the aggregation weight varies
        # with the neighbor's canonical position instead of being a uniform linear map (which folds
        # into `post_nn`). `gate_hidden_channels` is a small bottleneck — a function of a scalar gap
        # needs little width, so it is not a tuned hyperparameter (overridable via LDNA's **kwargs).
        gate_in_channels = 3
        self.gate_nn = nn.Sequential(
            nn.Linear(gate_in_channels, gate_hidden_channels),
            nn.ReLU(),
            nn.Linear(gate_hidden_channels, in_channels),
        )
        self.post_nn = MLP(in_channels=in_channels, hidden_channels=out_channels,
                           out_channels=out_channels, num_layers=num_post_layers)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.pre_nn.reset_parameters()
        self.post_nn.reset_parameters()
        # Start the gate at 1 so the conv begins as a plain sum aggregator. The first layer is
        # reset normally; zeroing the second layer makes gate_nn(feats) = 0 for any input, hence
        # `gate = 1 + gate_nn(feats) = 1` at init.
        self.gate_nn[0].reset_parameters()
        nn.init.zeros_(self.gate_nn[2].weight)
        nn.init.zeros_(self.gate_nn[2].bias)

    def forward(self, x, edge_index, edge_attr=None, nrank=None):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, nrank=nrank)

        return self.post_nn(out)

    def message(self, x_i, x_j, edge_attr=None, nrank_i=None, nrank_j=None):
        if edge_attr is not None:
            h = torch.cat([x_i, x_j, edge_attr], dim=-1)
        else:
            h = torch.cat([x_i, x_j], dim=-1)
        m = self.pre_nn(h)

        if nrank_i is not None:
            delta = nrank_j - nrank_i
            feats = torch.cat([delta, delta.abs(), torch.sign(delta)], dim=-1)
            gate = 1.0 + self.gate_nn(feats)

            return gate * m

        return m

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        out = scatter(inputs, index, dim=0, dim_size=dim_size, reduce='sum')

        return out
