import numpy as np
import torch
import torch.nn as nn
from Layers import EncoderLayer, DecoderLayer, PositionalEncoding
import random

class Transformer(nn.Module):
    """Main model in 'Attention is all you need'
    
    Args:
        bpe_size:   vocabulary size from byte pair encoding
        h:          number of heads
        d_model:    dimension of model
        p:          dropout probabolity 
        d_ff:       dimension of feed forward
        
    """
    def __init__(self, bpe_size, h, d_model, p, d_ff):
        super(Transformer, self).__init__()
        self.bpe_size = bpe_size
        self.word_emb = nn.Embedding(bpe_size, d_model, padding_idx=0)
        self.pos_emb = PositionalEncoding(d_model, p)
        self.encoder = nn.ModuleList([EncoderLayer(h, d_model, p, d_ff) for _ in range(6)]) 
        self.decoder = nn.ModuleList([DecoderLayer(h, d_model, p, d_ff) for _ in range(6)])
        self.generator = nn.Linear(d_model, bpe_size, bias=False)
        # tie weight between word embedding and generator 
        self.generator.weight = self.word_emb.weight
        self.logsoftmax = nn.LogSoftmax()
        # pre-save a mask to avoid future information in self-attentions in decoder
        # save as a buffer, otherwise will need to recreate it and move to GPU during every call
        mask = torch.ByteTensor(np.triu(np.ones((512,512)), k=1).astype('uint8'))
        self.register_buffer('mask', mask)
        
    def encode(self, src):
        context = self.word_emb(src)                                  # batch_size x len_src x d_model
        context = self.pos_emb(context)
        mask_src = src.data.eq(0).unsqueeze(1)                                       
        for _, layer in enumerate(self.encoder):                          
            context = layer(context, context, context, mask_src)      # batch_size x len_src x d_model
        return context, mask_src
    
    def decode(self, tgt, context, mask_src):
        out = self.word_emb(tgt)                                      # batch_size x len_tgt x d_model
        out = self.pos_emb(out)
        len_tgt = tgt.size(1)
        mask_tgt = tgt.data.eq(0).unsqueeze(1) + self.mask[:len_tgt, :len_tgt]
        mask_tgt = torch.gt(mask_tgt, 0)
        for _, layer in enumerate(self.decoder):
            out, coverage = layer(out, out, out, context, mask_tgt, mask_src)  # batch_size x len_tgt x d_model   
        out = self.generator(out)                                              # batch_size x len_tgt x bpe_size
        out = self.logsoftmax(out.view(-1, self.bpe_size))          
        return out, coverage
        
    def forward(self, src, tgt):
        """
        Inputs Shapes: 
            src: batch_size x len_src  
            tgt: batch_size x len_tgt
        
        Outputs Shapes:
            out:      batch_size*len_tgt x bpe_size
            coverage: batch_size x len_tgt x len_src
            
        """
        context, mask_src = self.encode(src)
        out, coverage = self.decode(tgt, context, mask_src)            
        return out, coverage

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        
        #src = [batch size, src len]
        #trg = [batch size, trg len]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time

        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        #encoder_outputs is all hidden states of the input sequence, back and forwards
        #hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(src)
                
        #first input to the decoder is the <sos> tokens
        input = trg[:,0].unsqueeze(0)
        
        for t in range(1, trg_len):
            
            #insert input token embedding, previous hidden state and all encoder hidden states
            #receive output tensor (predictions) and new hidden state
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            
            #place predictions in a tensor holding predictions for each token
            outputs[:,t] = output
            
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            #get the highest predicted token from our predictions
            top1 = output.argmax(1) 
            
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = trg[:,t] if teacher_force else top1
            input = input.unsqueeze(0)

        return outputs

class Seq2Seq_VFeat(nn.Module):
    def __init__(self, encoder_t, encoder_v, decoder, device):
        super().__init__()
        
        self.encoder_t = encoder_t
        self.encoder_v = encoder_v
        self.decoder = decoder
        self.device = device

    def forward(self, src_t, src_v, trg, teacher_forcing_ratio = 0.5):
        
        #src_t = [batch size, src len]
        #src_v = [batch size, vfeat len]
        #trg = [batch size, trg len]

        batch_size = src_t.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        enc_out_t, hidden = self.encoder_t(src_t)
        enc_out_v, hidden_v = self.encoder_v(src_v)
        input = trg[:,0].unsqueeze(0)
        
        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden, enc_out_t, enc_out_v)
            outputs[:,t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1) 
            input = trg[:,t] if teacher_force else top1
            input = input.unsqueeze(0)

        return outputs