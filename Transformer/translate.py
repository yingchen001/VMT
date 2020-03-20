# For data loading.
from torchtext import data, datasets
from data_process import ParallelDataset, Batch, subsequent_mask, batch_size_fn, read_corpus
import time
import torch
from torch.autograd import Variable
import numpy as np

from model import make_model
from optim import LabelSmoothing, NoamOpt
from loss import SimpleLossCompute
import argparse

from project import MyIterator

# Extra vocabulary sym  bols
pad_token = "<pad>"
unk_token = "<unk>"
bos_token = "<bos>"
eos_token = "<eos>"

extra_tokens = [pad_token, unk_token, bos_token, eos_token]

PAD = extra_tokens.index(pad_token)
UNK = extra_tokens.index(unk_token)
BOS = extra_tokens.index(bos_token)
EOS = extra_tokens.index(eos_token)

SRC = data.Field(sequential=True, use_vocab=False, include_lengths=True, batch_first=True,
                 pad_token=PAD, unk_token=UNK, init_token=BOS, eos_token=EOS, )
TGT = data.Field(sequential=True, use_vocab=False, include_lengths=True, batch_first=True,
                 pad_token=PAD, unk_token=UNK, init_token=BOS, eos_token=EOS, )
fields = (SRC, TGT)

path = 'vatex_len_40-train.t7'
max_src_len = 50
max_trg_len = 50
BATCH_SIZE = 128
device = 'cpu'

dataset = torch.load(path)
train_src, train_tgt = dataset['train_src'], dataset['train_tgt']
dev_src, dev_tgt = dataset['dev_src'], dataset['dev_tgt']


def filter_pred(example):
    if len(example.src) <= max_src_len and len(example.trg) <= max_trg_len:
        return True
    return False


train = ParallelDataset(train_src, train_tgt, fields=fields, filter_pred=filter_pred)
val = ParallelDataset(dev_src, dev_tgt, fields=fields, filter_pred=filter_pred)

MIN_FREQ = 1
SRC.build_vocab(train.src, min_freq=MIN_FREQ)
TGT.build_vocab(train.trg, min_freq=MIN_FREQ)

#########################################
# train_iter = MyIterator(train, batch_size=BATCH_SIZE, device=device,
#                         repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
#                         batch_size_fn=batch_size_fn, train=True)
valid_iter = MyIterator(val, batch_size=BATCH_SIZE, device=device,
                        repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                        batch_size_fn=batch_size_fn, train=False)

model = torch.load("final_model.pkl", map_location=torch.device('cpu'))


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)

    for i in range(max_len - 1):
        print(i)
        out = model.decode(memory, src_mask, Variable(ys), Variable(subsequent_mask(ys.size(1))).type_as(src.data))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys

# for batch in valid_iter:
#     break
# translate
for i, batch in enumerate(valid_iter):
    print(batch.src[0])
    src = batch.src[0][:1]
    print(src)
    break
    src_mask = (src != SRC.vocab.stoi[1]).unsqueeze(-2)

    out = greedy_decode(model, src, src_mask,
                        max_len=50, start_symbol=TGT.vocab.stoi[2])
    print("Translation:", end="\t")
    for i in range(1, out.size(1)):
        sym = TGT.vocab.itos[out[0, i]]
        if sym == 3: break
        print(sym, end=" ")
    print()
    print("Target:", end="\t")
    for i in range(1, batch.trg[0].size(1)):
        sym = TGT.vocab.itos[batch.trg[0][0, i]]
        if sym == 3: break
        print(sym, end=" ")
    print()
    break

