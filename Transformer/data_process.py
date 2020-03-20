import torch
import numpy as np
from torch.autograd import Variable
from torchtext import data, datasets

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8') # to get upper triangle, lower tri is all 0
    # get array of true/false: convert it to tensor and only get the entries which are 0
    return torch.from_numpy(subsequent_mask) == 0


class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            # remove the last col
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        # tgt != pad is to hide padding, return true or false;
        # unsqueeze is to add 1 dim in the second position from back
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


global max_src_in_batch, max_tgt_in_batch


def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


class ParallelDataset(data.Dataset):
    """Defines a custom dataset for machine translation."""
    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.src), len(ex.trg))

    def __init__(self, src_examples, trg_examples, fields, **kwargs):
        """Create a Translation Dataset given paths and fields.

        Arguments:
            path: Path to the data preprocessed with preprocess.py
            category: Whether the Dataset is for training or development
            fields: A tuple containing the fields that will be used for data
                in each language.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        if not isinstance(fields[0], (tuple, list)):
            if trg_examples is None:
                fields = [('src', fields[0])]
            else:
                fields = [('src', fields[0]), ('trg', fields[1])]

        examples = []
        if trg_examples is None:
            for src_line in src_examples:
                examples.append(data.Example.fromlist(
                    [src_line], fields))
        else:
            for src_line, trg_line in zip(src_examples, trg_examples):
                examples.append(data.Example.fromlist(
                    [src_line, trg_line], fields))

        super(ParallelDataset, self).__init__(examples, fields, **kwargs)


def rebatch(pad_idx, batch):
    "Fix order in torchtext to match ours"
    src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
    return Batch(src, trg, pad_idx)

def read_corpus(src_path, max_len, lower_case=False):
    print('Reading examples from {}..'.format(src_path))
    src_sents = []
    empty_lines, exceed_lines = 0, 0
    with open(src_path, encoding='utf8') as src_file:
        for idx, src_line in enumerate(src_file):
            if idx % 10000 == 0:
                print('  reading {} lines..'.format(idx))
            if src_line.strip() == '':  # remove empty lines
                empty_lines += 1
                continue
            if lower_case:  # check lower_case
                src_line = src_line.lower()

            src_words = src_line.strip().split()
            if max_len is not None and len(src_words) > max_len:
                exceed_lines += 1
                continue
            src_sents.append(src_words)
    print('Removed {} empty lines'.format(empty_lines),
          'and {} lines exceeding the length {}'.format(exceed_lines, max_len))
    print('Result: {} lines remained'.format(len(src_sents)))
    return src_sents