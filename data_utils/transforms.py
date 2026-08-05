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


class AddPosField(BaseTransform):
    # CIFAR10 + DGN: the superpixel image coordinates stand in for the Laplacian eigenvectors as the
    # directional field, which is what the DGN paper itself does on this dataset. Two reasons, both
    # measured here. The eigenvector path is not merely degenerate but wrong: CIFAR10's edge_index is
    # a DIRECTED 8-NN graph (20.05% of edges have no reverse, measured over the whole 45,000-graph
    # train split), so `add_eig_vec`'s `torch.linalg.eigh` — which reads only the lower triangle —
    # silently decomposes a matrix that is not the Laplacian it was handed. That leaves the field
    # unreliable rather than uniformly wrong: close to the correctly-symmetrized one on most graphs
    # and badly off on a large minority. And the field it approximates is ill-conditioned anyway:
    # the symmetrized lambda2-lambda1 gap has median 0.0280 against MNIST's 0.3328, with 7.8% of
    # graphs below 0.01 (n=500 graphs each). Coordinates have neither problem and cost a cast, so
    # like AddDepthField this rides the lazy transform chain with no precompute and no cache. No
    # centering or rescaling either: the field reaches DGNConv only as the per-edge difference
    # `pos_j - pos_i`, which kills any constant offset, and `aggregate` then L1-normalizes per node,
    # which kills the scale.
    def forward(self, data):
        data.eig_vec = data.pos.to(torch.float)
        return data


class ToUndirectedNoAttr(BaseTransform):
    # ogbg-code2 is edge-less: symmetrize connectivity by adding reverse edges only,
    # without producing any edge features.
    def forward(self, data):
        data.edge_index = torch.cat([data.edge_index, data.edge_index.flip(0)], dim=1)
        return data
