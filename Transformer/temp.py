import torch
import numpy as np
from torch.autograd import Variable
import math
V = 11
batch = 30
vbatch = 20

data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
data[:, 0] = 1
print(data)
tmp = torch.ones(30,20,10,5)
tmp = tmp.unsqueeze(-2)
print(tmp.shape)
pad = 0
print(tmp!=pad)

print(data.shape)
src = Variable(data, requires_grad=False) # src=data
tgt = Variable(data, requires_grad=False) # tgt=data

tgt = tgt[:, :-1]
print(tgt.shape)


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


tgt_mask = (tgt != pad).unsqueeze(-2)
print(tgt_mask)
print(tgt_mask.shape)
size = tgt.size(-1)
attn_shape = (1, size, size)
attn_shape
subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
mask2 = Variable((torch.from_numpy(subsequent_mask) == 0).type_as(tgt_mask.data))
mask2.shape
tgt_mask.shape
res = tgt_mask & mask2
res.shape
mask2
res[1]


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

x = torch.randint(0,10,[6,2,10])
query = x
key = x
d_k = query.size(-1)
d_k
scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
scores
scores.shape
attn_shape = (6,2,2)
subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
subsequent_mask = (torch.from_numpy(subsequent_mask) == 0)
subsequent_mask.shape
subsequent_mask

import torch.nn.functional as F
scores = scores.masked_fill(subsequent_mask == 0, -1e9)
p_attn = F.softmax(scores, dim = -1)
p_attn


key = torch.ones((18,8,6,64))
query = torch.ones((18,8,6,64))
key.transpose(-2, -1).shape
scores = torch.matmul(query, key.transpose(-2, -1))
scores.shape
src_mask == 0
scores = scores.masked_fill(src_mask == 0, -1e9)

for batch in valid_iter:
    break

src = batch.src[0]
src_mask = (src != SRC.vocab.stoi[1]).unsqueeze(-2)
max_len=50
start_symbol=TGT.vocab.stoi[2]
ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
ys
Variable(subsequent_mask(ys.size(1)))