import math

import torch.nn.functional as F
from torch import nn
from torch_geometric.nn.aggr import AttentionalAggregation
from torch_geometric.nn.conv import GENConv
from torch_geometric.nn.models import DeepGCNLayer, MLP
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.nn.pool import global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.utils import to_dense_batch


class DeeperGCN(nn.Module):
    def __init__(self, *,
                 channel_list=None, in_channels=None, hidden_channels=None, out_channels=None, num_layers=None,
                 edge_dim=None, node_encoder=None, edge_encoder=None, aggregator='softmax',
                 num_trans_layers=1, num_pred_heads=None, num_pred_layers=3, readout=None,
                 dropout=0.0, batch_norm=True, residual=False, act='relu', act_kwargs=None, **kwargs):
        super(DeeperGCN, self).__init__()

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
        self.aggregator = aggregator
        # DeeperGCN uses intrinsic 'res+' residual blocks; the shared `residual` flag is
        # accepted for interface parity but has no separate effect here.
        self.residual = residual

        if isinstance(dropout, float):
            dropout = [dropout] * (len(channel_list) - 1)
        if len(dropout) != len(channel_list) - 1:
            raise ValueError(f"Number of dropout values provided ({len(dropout)}) does not "
                             f"match the number of layers specified ({len(channel_list) - 1})")
        self.dropout = dropout

        self.layers = nn.ModuleList()
        for in_channels, out_channels, p_dropout in zip(channel_list[:-1], channel_list[1:], self.dropout):
            conv = GENConv(in_channels=in_channels, out_channels=out_channels, edge_dim=edge_dim,
                           aggr=aggregator, learn_t=True, learn_p=True, num_layers=num_trans_layers, **kwargs)
            norm = BatchNorm(out_channels) if batch_norm else None
            act = activation_resolver(act, **(act_kwargs or {}))

            self.layers.append(DeepGCNLayer(conv=conv, norm=norm, act=act, block='res+', dropout=p_dropout))

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
        for layer in self.layers:
            layer.reset_parameters()
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

        x = self.layers[0].conv(x, edge_index=edge_index, edge_attr=edge_attr)
        for layer in self.layers[1:]:
            x = layer(x, edge_index=edge_index, edge_attr=edge_attr)
        x = self.layers[0].act(self.layers[0].norm(x))
        x = F.dropout(x, self.dropout[0], training=self.training)

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
