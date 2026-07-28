import torch
from torch import nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.models import MLP
from torch_scatter import scatter


class LDNAConv(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim=None, num_pre_layers=1, num_post_layers=1,
                 gate_hidden_channels=32, gate_init_std=0.0, **kwargs):
        super(LDNAConv, self).__init__(aggr=None, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        # float() guards the YAML-1.1 gotcha where a bare `1e-1` in a config parses as a string;
        # rejecting negatives keeps a sign typo from silently falling through to the zero-init path.
        self.gate_init_std = float(gate_init_std)
        if self.gate_init_std < 0:
            raise ValueError(f"Argument `gate_init_std` must be >= 0, got {self.gate_init_std}")

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
        # `gate = 1 + gate_nn(feats) = 1` at init. A non-zero `gate_init_std` instead draws that
        # layer's weight from N(0, std²) (bias stays 0) so the gate starts slightly off identity:
        # on the synthetic expressiveness benchmarks an exact-identity start gives two
        # 1-WL-equivalent graphs identical embeddings at step 0, and the pairwise protocol's short
        # budget must first push the gate off identity before any separating signal exists. The
        # default 0.0 keeps every existing config byte-identical (and consumes no RNG).
        self.gate_nn[0].reset_parameters()
        if self.gate_init_std > 0:
            nn.init.normal_(self.gate_nn[2].weight, std=self.gate_init_std)
        else:
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
