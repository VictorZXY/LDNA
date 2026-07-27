# PyG-2.7 loader for the EXP / CEXP expressiveness datasets (Abboud et al., 2021).
#
# Each dataset ships as a single pickle (raw/GRAPHSAT.pkl) of 1200 graphs: 600 consecutive
# (2i, 2i+1) pairs that are 1-WL-indistinguishable yet carry opposite binary labels. The pickle was
# written with an old PyG whose `Data` kept its tensors directly in `__dict__`; under PyG 2.7
# attribute access on those objects raises, so `process` reaches the tensors through `__dict__` and
# re-wraps each graph into a fresh `Data`. Adapted from the reference `PlanarSATPairsDataset`
# (Abboud et al., GNN-RNI); the `__dict__` re-wrap and the PyG-2.7 idioms are the only changes.
import os
import pickle

import torch
from torch_geometric.data import Data, InMemoryDataset

NAME = "GRAPHSAT"


class PlanarSATPairsDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [NAME + ".pkl"]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        pass

    def process(self):
        raw = pickle.load(open(os.path.join(self.raw_dir, NAME + ".pkl"), "rb"))

        # Old-PyG `Data` stored its tensors in `__dict__`; re-wrap each graph into a fresh PyG-2.7
        # `Data`. `x` is the raw [n, 1] int64 node-type column (embedded by the encoder, not
        # one-hot), and `y` is the single graph label reshaped to [1] long so batching yields a
        # [batch] target for CrossEntropyLoss / argmax accuracy. The graphs carry no edge features.
        data_list = []
        for old in raw:
            store = old.__dict__
            data_list.append(Data(x=store['x'], edge_index=store['edge_index'],
                                   y=store['y'].view(-1).long()))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
