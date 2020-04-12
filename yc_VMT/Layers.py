import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class LayerNorm(nn.Module):
    """Applies layer normalization to last dimension

    Args:
        d: dimension of hidden units

    """
    def __init__(self, d):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(d), requires_grad=True)

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + 1e-6) + self.beta


class MultiHeadAttention(nn.Module):
    """Applies multi-head attentions to inputs (query, key, value)

    Args:
        h:       number of heads
        d_model: dimension of model
        p:       dropout probabolity  
        
    Params:
        fc_query:  FC layer to project query, d_model x (h x d_head)
        fc_key:    FC layer to project key,   d_model x (h x d_head)
        fc_value:  FC layer to project value, d_model x (h x d_head)
        fc_concat: FC layer to concat and project multiheads, d_model x (h x d_head)
        
    Inputs Shapes: 
        query: batch_size x len_query x d_model 
        key:   batch_size x len_key x d_model   
        value: batch_size x len_key x d_model
        mask:  batch_size x len_query x len_key or broadcastable 
        
    Outputs Shapes:
        out:      batch_size x len_query x d_model
        coverage: batch_size x len_query x len_key
        
    """
    
    def __init__(self, h, d_model, p):
        super(MultiHeadAttention, self).__init__()      
        self.h = h
        self.d = d_model
        self.d_head = d_model//h
        self.fc_query = nn.Linear(d_model, h*self.d_head, bias=False)
        self.fc_key = nn.Linear(d_model, h*self.d_head, bias=False)
        self.fc_value = nn.Linear(d_model, h*self.d_head, bias=False)
        self.fc_concat = nn.Linear(h*self.d_head, d_model, bias=False)
        self.sm = nn.Softmax()
        self.dropout = nn.Dropout(p)
        self.attn_dropout = nn.Dropout(p)
        self.layernorm = LayerNorm(d_model)
      
    def _prepare_proj(self, x):
        """Reshape the projectons to apply softmax on each head

        """
        b, l, d = x.size()
        return x.view(b, l, self.h, self.d_head).transpose(1,2).contiguous().view(b*self.h, l, self.d_head)
        
    def forward(self, query, key, value, mask):
        b, len_query = query.size(0), query.size(1)
        len_key = key.size(1)
        
        # project inputs to multi-heads
        proj_query = self.fc_query(query)       # batch_size x len_query x h*d_head
        proj_key = self.fc_key(key)             # batch_size x len_key x h*d_head
        proj_value = self.fc_value(value)       # batch_size x len_key x h*d_head
        
        # prepare the shape for applying softmax
        proj_query = self._prepare_proj(proj_query)  # batch_size*h x len_query x d_head
        proj_key = self._prepare_proj(key)           # batch_size*h x len_key x d_head
        proj_value = self._prepare_proj(value)       # batch_size*h x len_key x d_head
        
        # get dotproduct softmax attns for each head
        attns = torch.bmm(proj_query, proj_key.transpose(1,2))  # batch_size*h x len_query x len_key
        attns = attns / math.sqrt(self.d_head) 
        attns = attns.view(b, self.h, len_query, len_key) 
        attns = attns.masked_fill_(Variable(mask.unsqueeze(1)), -float('inf'))
        attns = self.sm(attns.view(-1, len_key))
        # return mean attention from all heads as coverage 
        coverage = torch.mean(attns.view(b, self.h, len_query, len_key), dim=1)
        attns = self.attn_dropout(attns)
        attns = attns.view(b*self.h, len_query, len_key)
        
        # apply attns on value
        out = torch.bmm(attns, proj_value)      # batch_size*h x len_query x d_head
        out = out.view(b, self.h, len_query, self.d_head).transpose(1,2).contiguous() 
        out = self.fc_concat(out.view(b, len_query, self.h*self.d_head))
        out = self.layernorm(query + self.dropout(out))   
        return out, coverage

    
class FeedForward(nn.Module):
    """Applies position-wise feed forward to inputs
    
    Args:
        d_model: dimension of model 
        d_ff:    dimension of feed forward
        p:       dropout probabolity 
        
    Params:
        fc_1: FC layer from d_model to d_ff
        fc_2: FC layer from d_ff to d_model
        
    Input Shapes:
        input: batch_size x len x d_model
        
    Output Shapes:
        out: batch_size x len x d_model

    """
    
    def __init__(self, d_model, d_ff, p):
        super(FeedForward, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.fc_1 = nn.Linear(d_model, d_ff)
        self.fc_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p)
        self.layernorm = LayerNorm(d_model)
        self.relu = nn.ReLU()
        
    def forward(self, input):
        out = self.dropout(self.fc_2(self.relu(self.fc_1(input))))
        out = self.layernorm(out + input)
        return out

    
class EncoderLayer(nn.Module):
    """Wraps multi-head attentions and position-wise feed forward into one encoder layer
    
    Args:
        h:       number of heads
        d_model: dimension of model
        p:       dropout probabolity 
        d_ff:    dimension of feed forward
        
    Params:
        multihead:    multi-head attentions layer
        feedforward:  feed forward layer
    
    Input Shapes:
        query: batch_size x len_query x d_model 
        key:   batch_size x len_key x d_model   
        value: batch_size x len_key x d_model
        mask:  batch_size x len_query x len_key or broadcastable 
    
    Output Shapes:
        out: batch_size x len_query x d_model

    """
    
    def __init__(self, h, d_model, p, d_ff):
        super(EncoderLayer, self).__init__()
        self.multihead = MultiHeadAttention(h, d_model, p)
        self.feedforward = FeedForward(d_model, d_ff, p)
    
    def forward(self, query, key, value, mask):
        out, _ = self.multihead(query, key, value, mask)
        out = self.feedforward(out)
        return out
    
    
class DecoderLayer(nn.Module):
    """Wraps multi-head attentions and position-wise feed forward into one layer of decoder
    
    Args:
        h:       number of heads
        d_model: dimension of model
        p:       dropout probabolity 
        d_ff:    dimension of feed forward
        
    Params:
        multihead_tgt:  multi-head self attentions layer
        multihead_src:  multi-head encoder-decoder attentions layer        
        feedforward:    feed forward layer
    
    Input Shapes:
        query:    batch_size x len_query x d_model 
        key:      batch_size x len_key x d_model   
        value:    batch_size x len_key x d_model
        context:  batch_size x len_src x d_model
        mask_tgt: batch_size x len_query x len_key or broadcastable 
        mask_src: batch_size x len_query x len_src or broadcastable 
    
    Output Shapes:
        out:      batch_size x len_query x d_model
        coverage: batch_size x len_query x len_key
        
    """    
    
    def __init__(self, h, d_model, p, d_ff):
        super(DecoderLayer, self).__init__()
        self.multihead_tgt = MultiHeadAttention(h, d_model, p)
        self.multihead_src = MultiHeadAttention(h, d_model, p)
        self.feedforward = FeedForward(d_model, d_ff, p)
    
    def forward(self, query, key, value, context, mask_tgt, mask_src):
        out, _ = self.multihead_tgt(query, key, value, mask_tgt)
        out, coverage = self.multihead_src(out, context, context, mask_src)
        out = self.feedforward(out)
        return out, coverage

class PositionalEncoding(nn.Module):
    """Adds positional embeddings to standard word embeddings 

    This matches the original TensorFlow implementation at https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py.
    
    Args:
        d_model: dimension of model
        p:       dropout probabolity  
        len_max: max seq length for pre-calculated positional embeddings
        
    Inputs Shapes: 
        word_emb: batch_size x len_seq x d_model 
        
    Outputs Shapes:
        out:   batch_size x len_seq x d_model
        
    """
    def __init__(self, d_model, p, len_max=512):
        # save a fixed positional embedding matrix up to len_max,
        # so that no need to recreate it everytime
        super(PositionalEncoding, self).__init__()
        position = torch.arange(0,len_max)  
        # print(position.dtype)                    
        num_timescales = d_model // 2
        log_timescale_increment = math.log(10000) / (num_timescales-1)
        inv_timescales = torch.exp((torch.arange(0, num_timescales) * -log_timescale_increment).type(torch.float))
        scaled_time = position.unsqueeze(1).type(torch.float) * inv_timescales.unsqueeze(0)
        pos_emb = torch.cat((torch.sin(scaled_time), torch.cos(scaled_time)), 1)
        # wrap in a buffer so that model can be moved to GPU
        self.register_buffer('pos_emb', pos_emb)
        self.dropout = nn.Dropout(p)
        
    def forward(self, word_emb):
        len_seq = word_emb.size(1)
        out = word_emb + Variable(self.pos_emb[:len_seq, :])
        out = self.dropout(out)
        return out

# Basic Seq2Seq Encoder and Decoder
class EncoderRNN(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim):
        super(EncoderRNN, self).__init__()

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, enc_hid_dim, num_layers=2, bidirectional=True)

    def forward(self, input):
        #input = [batch size, src_len]
        embedded = self.embedding(input).permute(1,0,2)
        try:
            output, hidden = self.lstm(embedded)
        except:
            print(embedded.shape)
        #  encoder RNNs fed through a linear layer
        batch_size = input.shape[0]
        # Make sure dec_hid_dim = 2* enc_hid_dim
        hidden_n = hidden[0].contiguous().view(2, 2, batch_size, -1).permute(0,2,1,3)
        hidden_n = hidden_n.contiguous().view(2,batch_size,-1)

        cell_n = hidden[1].contiguous().view(2, 2, batch_size, -1).permute(0,2,1,3)
        cell_n = cell_n.contiguous().view(2,batch_size,-1)
        #outputs = [src len, batch size, enc hid dim * 2]
        #hidden = [2, batch size, enc hid dim * 2]
        return output, (hidden_n, cell_n)

class EncoderRNN_VFeat(nn.Module):
    def __init__(self, input_dim, enc_hid_dim):
        super(EncoderRNN_VFeat, self).__init__()
        self.lstm = nn.LSTM(input_dim, enc_hid_dim, bidirectional=True)
    def forward(self, input):
        output = input.permute(1,0,2)
        # Shape (seq_len, batch, input_size)
        output, hidden = self.lstm(output)
        batch_size = input.shape[0]
        # Make sure dec_hid_dim = 2* enc_hid_dim
        hidden_n = hidden[0].permute(1,0,2).contiguous().view(1,batch_size,-1)
        cell_n = hidden[1].permute(1,0,2).contiguous().view(1,batch_size,-1)
        #outputs = [src len, batch size, enc hid dim * 2]
        #hidden = [1, batch size, enc_hid_dim * 2]
        return output, (hidden_n, cell_n)


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        
        self.attn = nn.Linear(enc_hid_dim + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)
        
    def forward(self, hidden, encoder_outputs):
        
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src len, batch size, enc hid dim]
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        #repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        energy = torch.cat((hidden, encoder_outputs), dim = 2)
        energy = self.attn(energy)
        energy = torch.tanh(energy)  #energy = [batch size, src len, dec hid dim]

        attention = self.v(energy).squeeze(2) #attention= [batch size, src len]
        
        return F.softmax(attention, dim=1)

class AttnDecoderRNN(nn.Module):

    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, attention, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.output_dim = output_dim
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.dropout_p = dropout_p
        self.attention = attention

        self.embedding = nn.Embedding(self.output_dim, self.emb_dim)
        self.dropout = nn.Dropout(self.dropout_p)
        self.lstm = nn.LSTM(self.enc_hid_dim + self.emb_dim, self.dec_hid_dim, num_layers=2)
        self.out = nn.Linear(self.dec_hid_dim, self.output_dim)

    def forward(self, input, hidden, enc_out_t):
        #hidden = ([*, batch size, dec hid dim], c_n)
        #enc_out_t = [src len, batch size, enc hid dim]
        
        batch_size = enc_out_t.shape[1]
        src_len = enc_out_t.shape[0]

        # input shape [1, batch]
        # input = input.unsqueeze(0) 

        embedded = self.embedding(input)
        embedded = self.dropout(embedded) #  [1, batch size, emb dim]
        attn_t = self.attention(hidden[0][0], enc_out_t)
        enc_out_t = enc_out_t.permute(1, 0, 2)
        attn_t_applied = torch.bmm(attn_t.unsqueeze(1), enc_out_t).permute(1, 0, 2) #[1, batch size, enc hid dim]

        rnn_input  = torch.cat((embedded, attn_t_applied), 2)
        output, hidden = self.lstm(rnn_input, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden

class AttnDecoderRNN_V(nn.Module):

    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, attention, dropout_p=0.1):
        super(AttnDecoderRNN_V, self).__init__()
        self.output_dim = output_dim
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.dropout_p = dropout_p
        self.attention = attention

        self.embedding = nn.Embedding(self.output_dim, self.emb_dim)
        self.dropout = nn.Dropout(self.dropout_p)
        self.lstm = nn.LSTM(self.enc_hid_dim * 2 + self.emb_dim, self.dec_hid_dim, num_layers=2)
        self.out = nn.Linear(self.dec_hid_dim, self.output_dim)

    def forward(self, input, hidden, enc_out_t, enc_out_v):
        #hidden = ([*, batch size, dec hid dim], c_n)
        #enc_out_t, enc_out_v = [src len, batch size, enc hid dim]
        assert enc_out_v.shape[2] == enc_out_v.shape[2]

        batch_size = enc_out_t.shape[1]
        src_len = enc_out_t.shape[0]

        # input shape [1, batch]
        # input = input.unsqueeze(0) 

        embedded = self.embedding(input)
        embedded = self.dropout(embedded) #  [1, batch size, emb dim]
        attn_t = self.attention(hidden[0][1], enc_out_t)
        attn_v = self.attention(hidden[0][0], enc_out_v)
        enc_out_t = enc_out_t.permute(1, 0, 2)
        enc_out_v = enc_out_v.permute(1, 0, 2)
        attn_t_applied = torch.bmm(attn_t.unsqueeze(1), enc_out_t).permute(1, 0, 2) #[1, batch size, enc hid dim]
        attn_v_applied = torch.bmm(attn_v.unsqueeze(1), enc_out_v).permute(1, 0, 2) #[1, batch size, enc hid dim]

        rnn_input  = torch.cat((embedded, attn_t_applied, attn_v_applied), 2)
        output, hidden = self.lstm(rnn_input, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden