import math

import torch.nn.functional as F
from torch import nn
from torch_geometric.nn.aggr import AttentionalAggregation
from torch_geometric.nn.models import MLP
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.nn.pool import global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.utils import to_dense_batch

from models import LDNAConv


class LDNA(nn.Module):
    def __init__(self, *,
                 channel_list=None, in_channels=None, hidden_channels=None, out_channels=None, num_layers=None,
                 edge_dim=None, node_encoder=None, edge_encoder=None, num_pre_layers=1, num_post_layers=1,
                 num_pred_heads=None, num_pred_layers=3, readout=None, dropout=0.0, batch_norm=True,
                 residual=False, act='relu', act_first=False, act_kwargs=None, **kwargs):
        super(LDNA, self).__init__()

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

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for in_channels, out_channels in zip(channel_list[:-1], channel_list[1:]):
            self.convs.append(LDNAConv(in_channels=in_channels, out_channels=out_channels, edge_dim=edge_dim,
                                      num_pre_layers=num_pre_layers, num_post_layers=num_post_layers, **kwargs))
            if batch_norm:
                self.batch_norms.append(BatchNorm(out_channels))
            else:
                self.batch_norms.append(None)

        # Per-layer residual connections. Where a layer changes width the skip is a
        # learnable linear projection; where it does not it is a parameter-free identity.
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

    def forward(self, x, edge_index, edge_attr, batch):
        if self.node_encoder is not None:
            x = self.node_encoder(x)
        if self.edge_encoder is not None:
            edge_attr = self.edge_encoder(edge_attr)

        for i, (conv, batch_norm, dropout) in enumerate(zip(self.convs, self.batch_norms, self.dropout)):
            x_res = x
            x = conv(x, edge_index=edge_index, edge_attr=edge_attr)

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
