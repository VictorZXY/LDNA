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


class AddDepthField(BaseTransform):
    # ogbg-code2 + DGN: the AST depth stands in for the Laplacian eigenvector as the directional
    # field. Every graph is a tree, so |depth_j - depth_i| = 1 on every symmetrized edge and the
    # gradient is exactly the parent/child orientation. Reads the raw, unclamped `node_depth`.
    def forward(self, data):
        data.eig_vec = data.node_depth.view(-1, 1).to(torch.float)
        return data


class ToUndirectedNoAttr(BaseTransform):
    # ogbg-code2 is edge-less: symmetrize connectivity by adding reverse edges only,
    # without producing any edge features.
    def forward(self, data):
        data.edge_index = torch.cat([data.edge_index, data.edge_index.flip(0)], dim=1)
        return data
