from ogb.graphproppred import Evaluator as _OGBEvaluator
from ogb.nodeproppred import Evaluator as _OGBNodeEvaluator
from sklearn.metrics import f1_score, roc_auc_score

from data_utils.code2 import decode_arr_to_seq


class Code2Evaluator:
    # Wraps the OGB ogbg-code2 F1 evaluator. The training loop works with per-position index
    # predictions, so this evaluator also carries `idx2vocab` and decodes a predicted index
    # matrix (num_graphs, max_seq_len) back into token sequences before scoring.
    def __init__(self, name='ogbg-code2', idx2vocab=None, **kwargs):
        self._e = _OGBEvaluator(name=name)
        self.eval_metric = self._e.eval_metric
        self.idx2vocab = idx2vocab

    def decode(self, mat):
        return [decode_arr_to_seq(mat[i], self.idx2vocab) for i in range(mat.size(0))]

    def eval(self, seq_ref, seq_pred):
        return self._e.eval({'seq_ref': seq_ref, 'seq_pred': seq_pred})


class MNISTEvaluator:
    def __init__(self, **kwargs):
        self.eval_metric = 'acc'

    def eval(self, input_dict):
        if not 'y_true' in input_dict:
            raise RuntimeError("Missing key of y_true")
        if not 'y_pred' in input_dict:
            raise RuntimeError("Missing key of y_pred")

        y_true, y_pred = input_dict['y_true'], input_dict['y_pred']

        """
            y_true: numpy ndarray or torch tensor of shape (num_graphs)
            y_pred: numpy ndarray or torch tensor of shape (num_graphs, num_classes=10)
        """
        
        y_pred = y_pred.argmax(dim=1)
        total_acc = y_pred.eq(y_true).sum().item()
        
        return {self.eval_metric: total_acc / len(y_true)}


class AccuracyEvaluator:
    # Generic argmax accuracy for single-label graph classification (EXP/CEXP). Mirrors
    # MNISTEvaluator's interface but is kept separate so each dataset keeps its own evaluator entry.
    def __init__(self, **kwargs):
        self.eval_metric = 'acc'

    def eval(self, input_dict):
        if not 'y_true' in input_dict:
            raise RuntimeError("Missing key of y_true")
        if not 'y_pred' in input_dict:
            raise RuntimeError("Missing key of y_pred")

        y_true, y_pred = input_dict['y_true'], input_dict['y_pred']

        """
            y_true: numpy ndarray or torch tensor of shape (num_graphs)
            y_pred: numpy ndarray or torch tensor of shape (num_graphs, num_classes)
        """

        # y_true arrives as (num_graphs,) at graph level but as (num_nodes, 1) from the node-level
        # datasets, where the bare `eq` would broadcast (N,) against (N, 1) into an N x N matrix —
        # a wrong accuracy on a small graph and an out-of-memory allocation on a large one. The
        # flatten is a no-op on the already 1-D EXP/CEXP targets.
        y_pred = y_pred.argmax(dim=1)
        total_acc = y_pred.eq(y_true.view(-1)).sum().item()

        return {self.eval_metric: total_acc / len(y_true)}


class OGBNodeEvaluator:
    # Wraps the OGB node-property-prediction evaluator (ogbn-arxiv and friends). Unlike its
    # graph-level counterpart it scores class INDICES, not logits, and requires both y_true and
    # y_pred shaped (num_nodes, 1), so the argmax and the reshape happen here rather than being
    # pushed onto the training loop. Mirrors how Code2Evaluator wraps the graph-level evaluator.
    def __init__(self, name='ogbn-arxiv', **kwargs):
        self._e = _OGBNodeEvaluator(name=name)
        self.eval_metric = self._e.eval_metric

    def eval(self, input_dict):
        if not 'y_true' in input_dict:
            raise RuntimeError("Missing key of y_true")
        if not 'y_pred' in input_dict:
            raise RuntimeError("Missing key of y_pred")

        y_true, y_pred = input_dict['y_true'], input_dict['y_pred']

        """
            y_true: numpy ndarray or torch tensor of shape (num_nodes) or (num_nodes, 1)
            y_pred: numpy ndarray or torch tensor of shape (num_nodes, num_classes)
        """

        y_true = y_true.detach().cpu().view(-1, 1)
        y_pred = y_pred.detach().cpu().argmax(dim=1).view(-1, 1)

        return self._e.eval({'y_true': y_true, 'y_pred': y_pred})


class ROCAUCEvaluator:
    # Binary node classification with a single logit per node (Questions). The split mask has
    # already removed every unscored node by the time the loop hands the tensors over, so the whole
    # input is scored.
    def __init__(self, **kwargs):
        self.eval_metric = 'rocauc'

    def eval(self, input_dict):
        if not 'y_true' in input_dict:
            raise RuntimeError("Missing key of y_true")
        if not 'y_pred' in input_dict:
            raise RuntimeError("Missing key of y_pred")

        y_true, y_pred = input_dict['y_true'], input_dict['y_pred']

        """
            y_true: numpy ndarray or torch tensor of shape (num_nodes) or (num_nodes, num_tasks=1)
            y_pred: numpy ndarray or torch tensor of shape (num_nodes, num_tasks=1)
        """

        y_true = y_true.detach().cpu().view(-1).numpy()
        y_pred = y_pred.detach().cpu().view(-1).numpy()

        # ROC-AUC is undefined when a split carries a single class. Raising here rather than
        # returning nan is deliberate: Logger tracks the best epoch with torch.max, which a nan
        # would silently propagate through, so the whole run would report nan instead of failing.
        if y_true.min() == y_true.max():
            raise ValueError(f"ROC-AUC is undefined for a split containing one class; got "
                             f"{len(y_true)} nodes all labelled {y_true[0]:g}")

        return {self.eval_metric: roc_auc_score(y_true, y_pred)}


class MicroF1Evaluator:
    # Multi-label node classification (PPI's 121 independent binary tasks). The model emits logits,
    # not probabilities, so the positive/negative boundary is 0 rather than 0.5.
    def __init__(self, **kwargs):
        self.eval_metric = 'f1'

    def eval(self, input_dict):
        if not 'y_true' in input_dict:
            raise RuntimeError("Missing key of y_true")
        if not 'y_pred' in input_dict:
            raise RuntimeError("Missing key of y_pred")

        y_true, y_pred = input_dict['y_true'], input_dict['y_pred']

        """
            y_true: numpy ndarray or torch tensor of shape (num_nodes, num_tasks)
            y_pred: numpy ndarray or torch tensor of shape (num_nodes, num_tasks)
        """

        y_true = y_true.detach().cpu().numpy()
        y_pred = (y_pred.detach().cpu() > 0).long().numpy()

        return {self.eval_metric: f1_score(y_true, y_pred, average='micro', zero_division=0)}


class ZINCEvaluator:
    def __init__(self, **kwargs):
        self.eval_metric = 'mae'

    def eval(self, input_dict):
        if not 'y_true' in input_dict:
            raise RuntimeError("Missing key of y_true")
        if not 'y_pred' in input_dict:
            raise RuntimeError("Missing key of y_pred")

        y_true, y_pred = input_dict['y_true'], input_dict['y_pred']

        """
            y_true: numpy ndarray or torch tensor of shape (num_graphs) or (num_graphs, num_tasks=1)
            y_pred: numpy ndarray or torch tensor of shape (num_graphs, num_tasks=1)
        """
        
        if y_true.dim() == 1:
            total_error = (y_pred.squeeze() - y_true).abs().sum().item()
        else:
            total_error = (y_pred - y_true).abs().sum().item()

        return {self.eval_metric: total_error / len(y_true)}
        