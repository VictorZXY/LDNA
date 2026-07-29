# --- torch>=2.6 load-compatibility shim --------------------------------------
# PyTorch 2.6 changed the default of ``torch.load(weights_only=...)`` from
# ``False`` to ``True``. OGB/PyG store their processed datasets as pickled
# objects that reference framework classes (e.g.
# ``torch_geometric.data.data.DataEdgeAttr``), which the ``weights_only=True``
# loader refuses to unpickle. Those caches are produced locally by trusted
# libraries, so we restore the pre-2.6 behaviour for all ``torch.load`` calls.
# The shim lives here because every ``torch.load`` of a dataset cache goes
# through this package; ``utils`` imports it via the resolver.
import torch as _torch

if not getattr(_torch.load, "_ldna_weights_only_patched", False):
    _ldna_orig_torch_load = _torch.load

    def _ldna_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _ldna_orig_torch_load(*args, **kwargs)

    _ldna_torch_load._ldna_weights_only_patched = True
    _torch.load = _ldna_torch_load
# -----------------------------------------------------------------------------

from .preprocess import add_eig_vec, add_eig_vecs, canonicalize_graph, canonicalize_graphs, degree_histogram, sort_graph, sort_graphs
from .transforms import RemoveEdgeAttr, UnsqueezeTargetDim

__all__ = [
    'add_eig_vec',
    'add_eig_vecs',
    'canonicalize_graph',
    'canonicalize_graphs',
    'degree_histogram',
    'sort_graph',
    'sort_graphs',
    'RemoveEdgeAttr',
    'UnsqueezeTargetDim'
]
