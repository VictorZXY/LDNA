# --- torch>=2.6 load-compatibility shim --------------------------------------
# PyTorch 2.6 changed the default of ``torch.load(weights_only=...)`` from
# ``False`` to ``True``. OGB/PyG store their processed datasets as pickled
# objects that reference framework classes (e.g.
# ``torch_geometric.data.data.DataEdgeAttr``), which the ``weights_only=True``
# loader refuses to unpickle. Those caches are produced locally by trusted
# libraries, so we restore the pre-2.6 behaviour for all ``torch.load`` calls.
import torch as _torch

if not getattr(_torch.load, "_ldna_weights_only_patched", False):
    _ldna_orig_torch_load = _torch.load

    def _ldna_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _ldna_orig_torch_load(*args, **kwargs)

    _ldna_torch_load._ldna_weights_only_patched = True
    _torch.load = _ldna_torch_load
# -----------------------------------------------------------------------------

from ._utils import sort_graph, sort_graphs
from .evaluator import ZINCEvaluator
from .logger import Logger
from .resolver import evaluator_resolver, loss_resolver, model_and_data_resolver
from .tee import Tee, tee_to_file
from .transforms import RemoveEdgeAttr, UnsqueezeTargetDim

__all__ = [
    'sort_graph',
    'sort_graphs',
    'ZINCEvaluator',
    'Logger',
    'evaluator_resolver',
    'loss_resolver',
    'model_and_data_resolver',
    'Tee',
    'tee_to_file',
    'RemoveEdgeAttr',
    'UnsqueezeTargetDim'
]
