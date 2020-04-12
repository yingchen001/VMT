import math
import random
import torch
import numpy as np
import pandas as pd
from torch.autograd import Variable

class Dataloader(object):
    """Class to Load Language Pairs and Make Batch
    """   
    def __init__(self, Filename, batch_size, src_lang='en', tgt_lang='zh', v_feat='i3d', max_len=40, cuda=False, volatile=False, sort=True):
        # Need to reload every time because memory error in pickle
        df = pd.read_csv(Filename)
        src_t = []
        src_v = []
        tgt = []
        nb_pairs = 0
        for index, row in df.iterrows():
            src_line, tgt_line = row[src_lang], row[tgt_lang]
            if src_line=='' and tgt_line=='':
                break            
            src_ids = list(map(int, src_line.strip().split()))
            # #Remove SOS and EOS for source 
            # src_ids = src_ids[1:-1]
            tgt_ids = list(map(int, tgt_line.strip().split()))
            if (0 in src_ids or 0 in tgt_ids):
                continue
            if len(src_ids)>0 and len(tgt_ids)>0:
                # Truncate instead of discarding the sentence
                src_t.append(src_ids if len(src_ids)<max_len+1 else src_ids[:max_len]+[3])
                if v_feat == 'i3d':
                    src_v.append(row['i3d_path'])
                tgt.append(tgt_ids if len(tgt_ids)<max_len+1 else tgt_ids[:max_len]+[3])
                nb_pairs += 1
        print('%d pairs are converted in the data' %nb_pairs)
        if sort:
            sorted_idx = sorted(range(nb_pairs), key=lambda i: len(src_t[i]))
        else:
            sorted_idx = [i for i in range(nb_pairs)]
        self.src_t = [src_t[i] for i in sorted_idx]
        self.src_v = [src_v[i] for i in sorted_idx] if src_v else []
        self.tgt = [tgt[i] for i in sorted_idx]
        self.batch_size = batch_size
        self.nb_pairs = nb_pairs
        self.nb_batches = math.ceil(nb_pairs/batch_size)
        self.v_feat = v_feat
        self.cuda = cuda
        self.volatile = volatile
        
    def __len__(self):
        return self.nb_batches  

    def _shuffle_index(self, n, m):
        """Yield indexes for shuffling a length n seq within every m elements"""
        indexes = []
        for i in range(n):
            indexes.append(i)
            if (i+1)%m ==0 or i==n-1:
                random.shuffle(indexes)
                for index in indexes:
                    yield index
                indexes = []
            
    def shuffle(self, m):
        """Shuffle the language pairs within every m elements
        
        This will make sure pairs in the same batch still have similr length.
        """
        shuffled_indexes = self._shuffle_index(self.nb_pairs, m)
        src_t, src_v, tgt = [], [], []
        for index in shuffled_indexes:
            src_t.append(self.src_t[index])
            tgt.append(self.tgt[index])
            if self.src_v:
                src_v.append(self.src_v[index])
        self.src_t = src_t
        self.src_v = src_v
        self.tgt = tgt
        
    def _wrap(self, sentences):
        """Pad sentences to same length and wrap into Variable"""
        max_size = max([len(s) for s in sentences])
        out = [s + [0]*(max_size-len(s)) for s in sentences]
        out = torch.LongTensor(out)
        if self.cuda:
            out = out.cuda()
        return Variable(out, volatile=self.volatile)
    
    def _v_feat_preprocess(self, paths):
        out = None
        if self.v_feat == 'i3d':
            # shape:(1, *, 1024)
            arrays = [np.load(path) for path in paths]
            # Pad zeros to make features have same size
            max_size = max([a.shape[1] for a in arrays])
            out = [np.pad(a,[(0, 0), (0, max_size-a.shape[1]), (0, 0)]) for a in arrays]
            out = torch.tensor(out).float()
            out = torch.squeeze(out, 1)
            if self.cuda:
                out = Variable(out, volatile=self.volatile).cuda()
        return out
        # TODO: preprocessing for raw videos or other encoder
        #As shapes of raw video or feature are not fixed, put them in list
        # elif self.v_feat == 'raw'
        # elif self.v_feat == 's3d'

    def __getitem__(self, i): 
        """Generate the i-th batch and wrap in Variable"""
        src_t_batch = self.src_t[i*self.batch_size:(i+1)*self.batch_size]
        src_v_batch = self.src_v[i*self.batch_size:(i+1)*self.batch_size]
        tgt_batch = self.tgt[i*self.batch_size:(i+1)*self.batch_size]

        return [self._wrap(src_t_batch), self._v_feat_preprocess(src_v_batch)], self._wrap(tgt_batch)