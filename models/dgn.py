import math

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn.aggr import AttentionalAggregation
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.models import MLP
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.nn.pool import global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.utils import degree, to_dense_batch
from torch_scatter import scatter

# Floating-point guard on the L1 normalizer of the field, matching the DGN reference.
EPS = 1e-8


class DGNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim=None,
                 aggregators=['mean', 'max', 'min', 'dir1-dx', 'dir1-av'],
                 scalers=['identity', 'amplification', 'attenuation'], deg=None,
                 num_pre_layers=1, num_post_layers=1, **kwargs):
        super(DGNConv, self).__init__(aggr=None, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.scalers = scalers

        # `dirK-av` (directional smoothing) and `dirK-dx` (directional derivative) weight the
        # messages by the gradient of eigenvector column K-1 of `eig_vec`. They are the reason
        # aggregation is written out in `aggregate()` instead of being delegated to PyG's
        # `DegreeScalerAggregation`: an `Aggregation` module never sees per-edge data, and
        # `dirK-dx` additionally needs the layer-input node feature.
        self.aggregators = aggregators
        self.dir_indices = {}
        for aggr in aggregators:
            if aggr.startswith('dir'):
                head, _, mode = aggr.partition('-')
                if mode not in ('av', 'dx') or not head[3:].isdigit():
                    raise ValueError(f"Could not resolve aggregator '{aggr}'")
                self.dir_indices[aggr] = int(head[3:]) - 1
            elif aggr not in ('sum', 'mean', 'min', 'max', 'var', 'std'):
                raise ValueError(f"Could not resolve aggregator '{aggr}'")

        # Degree scalers, identical to PNA's. `avg_deg_log` is the mean log in-degree of the
        # training split, derived from the same in-degree histogram `utils/resolver.py` already
        # builds for PNA, so the two baselines scale their aggregations the same way. Only
        # `amplification` / `attenuation` read it, hence `deg` is optional.
        if deg is not None:
            deg = deg.to(torch.float)
            avg_deg_log = ((torch.arange(deg.numel()) + 1).log() * deg).sum() / deg.sum()
        elif 'amplification' in scalers or 'attenuation' in scalers:
            raise ValueError("Argument `deg` must be given for the "
                             "`amplification` and `attenuation` scalers")
        else:
            avg_deg_log = torch.zeros(())
        self.register_buffer('avg_deg_log', avg_deg_log)

        if edge_dim is not None:
            self.pre_nn = MLP(in_channels=2 * in_channels + edge_dim, hidden_channels=in_channels,
                              out_channels=in_channels, num_layers=num_pre_layers)
        else:
            self.pre_nn = MLP(in_channels=2 * in_channels, hidden_channels=in_channels,
                              out_channels=in_channels, num_layers=num_pre_layers)
        self.post_nn = MLP(in_channels=(len(aggregators) * len(scalers) + 1) * in_channels,
                           hidden_channels=out_channels, out_channels=out_channels,
                           num_layers=num_post_layers)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.pre_nn.reset_parameters()
        self.post_nn.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None, eig_vec=None):
        if self.dir_indices and eig_vec is None:
            raise ValueError(f"Argument `eig_vec` must be given for the directional "
                             f"aggregators {sorted(self.dir_indices)}")

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, eig_vec=eig_vec)
        out = torch.cat([x, out], dim=-1)

        return self.post_nn(out)

    def message(self, x_i, x_j, edge_attr=None, eig_vec_i=None, eig_vec_j=None):
        if edge_attr is not None:
            h = torch.cat([x_i, x_j, edge_attr], dim=-1)
        else:
            h = torch.cat([x_i, x_j], dim=-1)
        m = self.pre_nn(h)

        if eig_vec_i is not None:
            # Carry the per-edge field F_ij = phi_j - phi_i (source minus target) alongside the
            # message: `aggregate()` is the only place it is consumed and PyG routes nothing else
            # per-edge. The message keeps width `in_channels`, so the two split cleanly again.
            return torch.cat([m, eig_vec_j - eig_vec_i], dim=-1)

        return m

    def aggregate(self, inputs, index, dim_size=None, x=None):
        m, field = inputs[..., :self.in_channels], inputs[..., self.in_channels:]

        # Row-wise L1 normalizer of the field, one value per node and eigenvector column;
        # `denom[index]` broadcasts it back onto the edges. Isolated nodes normalize by EPS alone
        # and scatter no messages, so the quotient never turns into a NaN.
        if self.dir_indices:
            denom = scatter(field.abs(), index, dim=0, dim_size=dim_size, reduce='sum') + EPS

        outs = []
        for aggr in self.aggregators:
            if aggr in self.dir_indices:
                eig_idx = self.dir_indices[aggr]
                z = denom[index][..., eig_idx:eig_idx + 1]
                if aggr.endswith('av'):
                    # Absolute value taken on the field: the weights are a non-negative partition
                    # of unity, so the result is invariant to the eigenvector's arbitrary sign.
                    weight = field[..., eig_idx:eig_idx + 1].abs() / z
                    out = scatter(weight * m, index, dim=0, dim_size=dim_size, reduce='sum')
                else:
                    # Signed weights, and the subtrahend is the layer-input feature `x_i` rather
                    # than a message (the asymmetry of the reference implementation, and why
                    # `pre_nn` maps in_channels -> in_channels). Here the absolute value is taken
                    # on the aggregated output, which is what makes `dx` sign-invariant.
                    weight = field[..., eig_idx:eig_idx + 1] / z
                    num = scatter(weight * m, index, dim=0, dim_size=dim_size, reduce='sum')
                    weight_sum = scatter(weight, index, dim=0, dim_size=dim_size, reduce='sum')
                    out = (num - weight_sum * x).abs()
            elif aggr in ('var', 'std'):
                mean = scatter(m, index, dim=0, dim_size=dim_size, reduce='mean')
                mean_sq = scatter(m * m, index, dim=0, dim_size=dim_size, reduce='mean')
                out = torch.relu(mean_sq - mean * mean)
                if aggr == 'std':
                    out = torch.sqrt(out + EPS)
            else:
                out = scatter(m, index, dim=0, dim_size=dim_size, reduce=aggr)
            outs.append(out)
        out = torch.cat(outs, dim=-1)

        deg = degree(index, num_nodes=dim_size, dtype=out.dtype).view(-1, 1)

        outs = []
        for scaler in self.scalers:
            if scaler == 'identity':
                outs.append(out)
            elif scaler == 'amplification':
                outs.append(out * (torch.log(deg + 1) / self.avg_deg_log))
            elif scaler == 'attenuation':
                # Clamp the degree to one so isolated nodes do not divide by zero.
                outs.append(out * (self.avg_deg_log / torch.log(deg.clamp(min=1) + 1)))
            else:
                raise ValueError(f"Could not resolve scaler '{scaler}'")

        return torch.cat(outs, dim=-1)


class DGN(nn.Module):
    def __init__(self, *,
                 channel_list=None, in_channels=None, hidden_channels=None, out_channels=None, num_layers=None,
                 edge_dim=None, node_encoder=None, edge_encoder=None,
                 aggregators=['mean', 'max', 'min', 'dir1-dx', 'dir1-av'],
                 scalers=['identity', 'amplification', 'attenuation'], deg=None, num_eig_vec=1,
                 num_pre_layers=1, num_post_layers=1, num_pred_heads=None, num_pred_layers=3, readout=None,
                 dropout=0.0, batch_norm=True, residual=False,
                 act='relu', act_first=False, act_kwargs=None, **kwargs):
        super(DGN, self).__init__()

        if in_channels is not None:
            if num_layers is None:
                raise ValueError("Argument `num_layers` must be given")
            if num_layers > 1 and hidden_channels is None:
                raise ValueError(f"Argument `hidden_channels` must be given for `num_layers={num_layers}`")
            if out_channels is None:
                raise ValueError("Argument `out_channels` must be given")
            channel_list = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]
        assert isinstance(channel_list, (tuple, list))
        assert len(channel_list) >= 2
        self.channel_list = channel_list

        self.node_encoder = node_encoder
        self.edge_encoder = edge_encoder
        self.aggregators = aggregators
        self.scalers = scalers

        # `dirK-*` reads eigenvector column K-1 of the field the resolver attaches to the data, so
        # the requested aggregators have to fit the number of columns it carries. Checked here so a
        # mismatch between the config and the resolver-side precompute fails at build time rather
        # than in the first forward; `DGN` makes no other use of `num_eig_vec`.
        self.num_eig_vec = num_eig_vec
        for aggr in aggregators:
            if aggr.startswith('dir'):
                eig_idx = int(aggr.partition('-')[0][3:]) - 1
                if eig_idx >= num_eig_vec:
                    raise ValueError(f"Aggregator '{aggr}' reads eigenvector column {eig_idx}, but "
                                     f"only `num_eig_vec={num_eig_vec}` columns are provided")

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for in_channels, out_channels in zip(channel_list[:-1], channel_list[1:]):
            self.convs.append(DGNConv(in_channels=in_channels, out_channels=out_channels, edge_dim=edge_dim,
                                      aggregators=aggregators, scalers=scalers, deg=deg,
                                      num_pre_layers=num_pre_layers, num_post_layers=num_post_layers, **kwargs))
            if batch_norm:
                self.batch_norms.append(BatchNorm(out_channels))
            else:
                self.batch_norms.append(None)

        # Per-layer residual connections: identity skip where widths match, learnable
        # linear projection where the layer changes width (e.g. the 128 -> hidden first layer).
        self.residual = residual
        self.res_projs = nn.ModuleList()
        if residual:
            for in_channels, out_channels in zip(channel_list[:-1], channel_list[1:]):
                self.res_projs.append(nn.Identity() if in_channels == out_channels
                                      else nn.Linear(in_channels, out_channels))

        self.act = activation_resolver(act, **(act_kwargs or {}))
        self.act_first = act_first

        if isinstance(dropout, float):
            dropout = [dropout] * (len(channel_list) - 1)
        if len(dropout) != len(channel_list) - 1:
            raise ValueError(f"Number of dropout values provided ({len(dropout)}) does not "
                             f"match the number of layers specified ({len(channel_list) - 1})")
        self.dropout = dropout

        self.readout = readout
        if readout == 'gru':
            self.readout_gru = nn.GRU(input_size=out_channels, hidden_size=out_channels, batch_first=True)
        elif readout == 'attention':
            out_channels = channel_list[-1]
            gate_nn = MLP(in_channels=out_channels, hidden_channels=out_channels, out_channels=1, num_layers=2)
            self.readout_attention = AttentionalAggregation(gate_nn=gate_nn)

        self.num_pred_heads = num_pred_heads
        if num_pred_heads:
            out_channels = channel_list[-1]
            if num_pred_heads > out_channels:
                raise ValueError(f"Number of output channels ({out_channels}) must be "
                                 f"greater than the number of prediction heads ({num_pred_heads})")
            num_proj_layers = min(num_pred_layers, math.ceil(math.log2(out_channels / num_pred_heads)))
            pred_channel_list = [out_channels // (2 ** l) for l in range(num_proj_layers)] \
                                + [num_pred_heads] * (num_pred_layers - num_proj_layers + 1)
            self.pred_nn = MLP(channel_list=pred_channel_list)

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for batch_norm in self.batch_norms:
            if batch_norm is not None:
                batch_norm.reset_parameters()
        for res_proj in self.res_projs:
            if hasattr(res_proj, 'reset_parameters'):
                res_proj.reset_parameters()
        if self.readout == 'gru':
            self.readout_gru.reset_parameters()
        elif self.readout == 'attention':
            self.readout_attention.reset_parameters()
        if self.num_pred_heads:
            self.pred_nn.reset_parameters()

    def forward(self, x, edge_index, edge_attr, batch, eig_vec=None):
        if self.node_encoder is not None:
            x = self.node_encoder(x)
        if self.edge_encoder is not None:
            edge_attr = self.edge_encoder(edge_attr)

        for i, (conv, batch_norm, dropout) in enumerate(zip(self.convs, self.batch_norms, self.dropout)):
            x_res = x
            x = conv(x, edge_index=edge_index, edge_attr=edge_attr, eig_vec=eig_vec)

            if self.act is not None and self.act_first:
                x = self.act(x)
            if batch_norm is not None:
                x = batch_norm(x)
            if self.act is not None and not self.act_first:
                x = self.act(x)

            x = F.dropout(x, p=dropout, training=self.training)

            if self.residual:
                x = x + self.res_projs[i](x_res)

        if self.readout == 'gru':
            x_dense, mask = to_dense_batch(x, batch)
            lengths = mask.sum(dim=1).cpu()
            x_packed = nn.utils.rnn.pack_padded_sequence(x_dense, lengths, batch_first=True, enforce_sorted=False)
            _, h_g = self.readout_gru(x_packed)
            x = h_g[0]
        elif self.readout == 'add':
            x = global_add_pool(x, batch)
        elif self.readout == 'max':
            x = global_max_pool(x, batch)
        elif self.readout == 'mean':
            x = global_mean_pool(x, batch)
        elif self.readout == 'attention':
            x = self.readout_attention(x, batch)

        if self.num_pred_heads:
            x = self.pred_nn(x)

        return x
