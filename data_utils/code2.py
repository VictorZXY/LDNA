# Vocabulary and target-sequence utilities for the ogbg-code2 task.
#
# The four functions below (get_vocab_mapping, encode_seq_to_arr, encode_y_to_arr,
# decode_arr_to_seq) are ported VERBATIM from the OGB reference implementation:
#   https://github.com/snap-stanford/ogb/blob/master/examples/graphproppred/code2/utils.py
# They turn the variable-length method-name token sequence stored in ``data.y`` into a
# fixed-length index array (``data.y_arr``) for the per-position classifier head, and turn a
# predicted index array back into a token sequence for the OGB F1 evaluator. The AST node
# encoder and edge-augmentation helpers from that same file are intentionally NOT ported: the
# LDNA code2 pipeline is edge-less and uses a 1-argument encoder (models/code2.py).
import numpy as np
import torch


def get_vocab_mapping(seq_list, num_vocab):
    '''
        Input:
            seq_list: a list of sequences
            num_vocab: vocabulary size
        Output:
            vocab2idx:
                A dictionary that maps vocabulary into integer index.
                Additioanlly, we also index '__UNK__' and '__EOS__'
                '__UNK__' : out-of-vocabulary term
                '__EOS__' : end-of-sentence

            idx2vocab:
                A list that maps idx to actual vocabulary.

    '''

    vocab_cnt = {}
    vocab_list = []
    for seq in seq_list:
        for w in seq:
            if w in vocab_cnt:
                vocab_cnt[w] += 1
            else:
                vocab_cnt[w] = 1
                vocab_list.append(w)

    cnt_list = np.array([vocab_cnt[w] for w in vocab_list])
    topvocab = np.argsort(-cnt_list, kind = 'stable')[:num_vocab]

    print('Coverage of top {} vocabulary:'.format(num_vocab))
    print(float(np.sum(cnt_list[topvocab]))/np.sum(cnt_list))

    vocab2idx = {vocab_list[vocab_idx]: idx for idx, vocab_idx in enumerate(topvocab)}
    idx2vocab = [vocab_list[vocab_idx] for vocab_idx in topvocab]

    # print(topvocab)
    # print([vocab_list[v] for v in topvocab[:10]])
    # print([vocab_list[v] for v in topvocab[-10:]])

    vocab2idx['__UNK__'] = num_vocab
    idx2vocab.append('__UNK__')

    vocab2idx['__EOS__'] = num_vocab + 1
    idx2vocab.append('__EOS__')

    # test the correspondence between vocab2idx and idx2vocab
    for idx, vocab in enumerate(idx2vocab):
        assert(idx == vocab2idx[vocab])

    # test that the idx of '__EOS__' is len(idx2vocab) - 1.
    # This fact will be used in decode_arr_to_seq, when finding __EOS__
    assert(vocab2idx['__EOS__'] == len(idx2vocab) - 1)

    return vocab2idx, idx2vocab


def encode_y_to_arr(data, vocab2idx, max_seq_len):
    '''
    Input:
        data: PyG graph object
        output: add y_arr to data
    '''

    # PyG >= 1.5.0
    seq = data.y

    # PyG = 1.4.3
    # seq = data.y[0]

    data.y_arr = encode_seq_to_arr(seq, vocab2idx, max_seq_len)

    return data


def encode_seq_to_arr(seq, vocab2idx, max_seq_len):
    '''
    Input:
        seq: A list of words
        output: add y_arr (torch.Tensor)
    '''

    augmented_seq = seq[:max_seq_len] + ['__EOS__'] * max(0, max_seq_len - len(seq))
    return torch.tensor([[vocab2idx[w] if w in vocab2idx else vocab2idx['__UNK__'] for w in augmented_seq]], dtype = torch.long)


def decode_arr_to_seq(arr, idx2vocab):
    '''
        Input: torch 1d array: y_arr
        Output: a sequence of words.
    '''


    eos_idx_list = torch.nonzero(arr == len(idx2vocab) - 1, as_tuple=False) # find the position of __EOS__ (the last vocab in idx2vocab)
    if len(eos_idx_list) > 0:
        clippted_arr = arr[: torch.min(eos_idx_list)] # find the smallest __EOS__
    else:
        clippted_arr = arr

    return list(map(lambda x: idx2vocab[x], clippted_arr.cpu()))
