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
        if i % opt.display_freq == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                    (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens


class MyIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b

            self.batches = pool(self.data(), self.random_shuffler)

        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size,
                                self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))


def rebatch(pad_idx, batch):
    "Fix order in torchtext to match ours"
    # ????
    src, trg = batch.src[0], batch.trg[0]
    return Batch(src, trg, pad_idx)


# load
def load_data(path, max_src_len, max_trg_len, BATCH_SIZE, device):

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

    # path = 'small-train.t7'
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
    train_iter = MyIterator(train, batch_size=BATCH_SIZE, device=device,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=True)
    valid_iter = MyIterator(val, batch_size=BATCH_SIZE, device=device,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=False)
    return SRC, TGT, train_iter, valid_iter


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


def main(opt):
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print("Use GPU...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    SRC, TGT, train_iter, valid_iter = load_data(opt.data_path, max_src_len=opt.max_src_seq_len, max_trg_len=opt.max_tgt_seq_len, BATCH_SIZE=opt.batch_size, device=device)
    pad_idx = TGT.vocab.stoi[1]
    model = make_model(len(SRC.vocab), len(TGT.vocab), N=opt.n_layers, d_model=opt.d_model, d_ff=opt.d_ff, h=opt.n_heads, dropout=opt.dropout).to(device)
    criterion = LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1).to(device)
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, opt.n_warp_steps,
                        torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.98), eps=1e-9))
    for epoch in range(opt.epochs):
        print('epoch:', epoch)
        model.train()
        run_epoch((rebatch(pad_idx, b) for b in train_iter),
                  model,
                  SimpleLossCompute(model.generator, criterion, model_opt))
        model.eval()
        run_epoch((rebatch(pad_idx, b) for b in valid_iter), model,
                  SimpleLossCompute(model.generator, criterion, opt=None))

    torch.save(model, opt.save_model)


# ################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training Hyperparams')
    parser.add_argument('-data_path', default='vatex_len_40-train.t7', help='Path to the preprocessed data')

    # network params
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_ff', type=int, default=2048)
    parser.add_argument('-n_heads', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-dropout', type=float, default=0.1)

    # training params
    parser.add_argument('-lr', type=float, default=0.02)
    parser.add_argument('-epochs', type=int, default=1)
    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-max_src_seq_len', type=int, default=50)
    parser.add_argument('-max_tgt_seq_len', type=int, default=50)
    # parser.add_argument('-max_grad_norm', type=float, default=None)
    parser.add_argument('-n_warp_steps', type=int, default=4000)
    parser.add_argument('-display_freq', type=int, default=100)
    # parser.add_argument('-log', default=None)
    # parser.add_argument('-model_path', type=str, default='None')
    parser.add_argument('-save_model', type=str, default='final_model.pkl')

    opt = parser.parse_args()
    print(opt)
    main(opt)





