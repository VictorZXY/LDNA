from ogb.graphproppred import Evaluator as _OGBEvaluator

from utils.code2 import decode_arr_to_seq


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
        