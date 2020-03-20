import time
import torch
from torch.autograd import Variable
import numpy as np

from model import make_model
from optim import LabelSmoothing, NoamOpt
from data_process import Batch, subsequent_mask
from loss import SimpleLossCompute

import copy

use_cuda = torch.cuda.is_available()

def run_epoch(data_iter, model, loss_compute):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg,
                            batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                    (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens


# simple example of generating the input data
def data_gen(source, target, batch):
    "Generate random data for a src-tgt copy task."
    for i in range(0, len(source) - batch, batch):

        size = min(batch, len(source) - i)
        src = source[i:i+size]
        tgt = target[i:i+size]

        src = torch.from_numpy(np.array(src)).type(torch.LongTensor)
        tgt = torch.from_numpy(np.array(tgt)).type(torch.LongTensor)

        src = Variable(src, requires_grad=False)
        tgt = Variable(tgt, requires_grad=False)

        if use_cuda:
            src = src.cuda()
            tgt = tgt.cuda()

        # print(src.shape)
        # print(tgt.shape)

        yield Batch(src, tgt, 0)


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(memory, src_mask, Variable(ys), Variable(subsequent_mask(ys.size(1)).type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys


def add_padding(source, dict, max_len):
    for i in range(0, len(source), 1):
        source[i].insert(0, dict['<bos>'])
        source[i].append(dict['<eos>'])
        source[i].extend([dict['<pad>']] * (max_len+2 - len(source[i])))
    return source


if __name__ == '__main__':
    data = torch.load("small-train.t7")
    v_src = len(data['src_dict'])
    v_tgt = len(data['tgt_dict'])

    source = data['train_src']
    target = data['train_tgt']
    val_source = data['dev_src']
    val_target = data['dev_tgt']

    max_len_src = len(max(source, key=len))
    max_len_tgt = len(max(target, key=len))
    max_len_val_src = len(max(val_source, key=len))
    max_len_val_tgt = len(max(val_target, key=len))

    source = add_padding(source, data['src_dict'], max_len_src)
    target = add_padding(target, data['tgt_dict'], max_len_tgt)
    val_source = add_padding(val_source, data['src_dict'], max_len_val_src)
    val_target = add_padding(val_target, data['tgt_dict'], max_len_val_tgt)

    criterion = LabelSmoothing(size=v_tgt, padding_idx=0, smoothing=0.0)
    model = make_model(v_src, v_tgt, N=2)

    if use_cuda:
        model = model.cuda()
    # model.src_embed[0] -> Embedding layer
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
                        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    for epoch in range(2):
        print(epoch)
        model.train()
        train = data_gen(source, target, 10)

        run_epoch(train, model,
                  SimpleLossCompute(model.generator, criterion, model_opt))

        model.eval()
        eval = data_gen(val_source, val_target, 10)
        print(run_epoch(eval, model,
                        SimpleLossCompute(model.generator, criterion, None)))

# # test data
# model.eval()
# src = Variable(torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]))
# src_mask = Variable(torch.ones(1, 1, 10))
# print(greedy_decode(model, src, src_mask, max_len=10, start_symbol=1))
