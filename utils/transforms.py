import torch
from torch_geometric.transforms import BaseTransform


class RemoveEdgeAttr(BaseTransform):
    def forward(self, data):
        data.edge_attr = None
        return data


class UnsqueezeTargetDim(BaseTransform):
    def forward(self, data):
        data.y = data.y.unsqueeze(dim=1)
        return data


class AddDepthToX(BaseTransform):
    # ogbg-code2: fold the AST node depth into x as column 2 so the 1-argument
    # ASTNodeEncoder can read [type, attr, depth] without a separate depth argument.
    def forward(self, data):
        data.x = torch.cat([data.x, data.node_depth.view(-1, 1)], dim=1)
        return data


class ToUndirectedNoAttr(BaseTransform):
    # ogbg-code2 is edge-less: symmetrize connectivity by adding reverse edges only,
    # without producing any edge features.
    def forward(self, data):
        data.edge_index = torch.cat([data.edge_index, data.edge_index.flip(0)], dim=1)
        return data
