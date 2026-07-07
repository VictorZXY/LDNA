from torch_geometric.transforms import BaseTransform


class RemoveEdgeAttr(BaseTransform):
    def forward(self, data):
        data.edge_attr = None
        return data


class UnsqueezeTargetDim(BaseTransform):
    def forward(self, data):
        data.y = data.y.unsqueeze(dim=1)
        return data
