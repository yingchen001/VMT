import os
import math
import pandas as pd
# import seaborn as sns
import torch
from queue import Queue
from torch.autograd import Variable

class Node(object):
    def __init__(self, hidden, previous_node, decoder_input, log_prob, length):
        self.hidden = hidden
        self.previous_node = previous_node
        self.decoder_input = decoder_input
        self.log_prob = log_prob
        self.length = length

class Translator(object):
    """Class to generate translations from beam search and to plot attention heatmap.
    
    Args:
        model:          pre-trained translation model with encode and decode method
        beam_size:      number of beams for beam search
        alpha:          low alpha means high penalty on hypothesis length
        beta:           low beta means low penalty on hypothesis coverage
        max_len:        max length for hypothesis
              
    """
    def __init__(self, model, max_len=40, beam_size=5, alpha=0.1, beta=0.3):
        self.model = model
        self.beam_size = beam_size
        self.alpha = alpha
        self.beta = beta
        self.max_len = max_len
        self.model.eval()

    def translate(self, src_id):
        logLikelihoods = []
        preds = []
        atten_probs = []
        coverage_penalties = []
        beam_size = self.beam_size
        remaining_beams = self.beam_size
        BOS_id = 2
        EOS_id = 3
        
        # generate context from source
#         src_id = self.word2id(source)
#         src_pieces = [self.id2word(str(i)) for i in src_id]
        with torch.no_grad():
            src_id = Variable(torch.LongTensor(src_id).unsqueeze(0).cuda(), volatile=True)
            input_length = src_id.shape[1]
            enc_out, hidden = self.model.encoder(src_id)

            # predict the first word
            decoder_input = Variable(torch.LongTensor([BOS_id]).unsqueeze(0).cuda())
            out, hidden = self.model.decoder(decoder_input, hidden, enc_out)
            scores, scores_id = out.view(-1).topk(beam_size)
            node = Node(hidden, None, decoder_input, 0, 1)
            q = Queue()
            q.put(node)
            end_nodes = []
            tgt_id = []
            while not q.empty():
                candidates = []
                # level traversal
                for _ in range(q.qsize()):
                    node = q.get()
                    decoder_input = node.decoder_input
                    decoder_hidden = node.hidden
                    if decoder_input.item() == EOS_id or node.length >= self.max_len:
                        end_nodes.append(node)
                        continue
                    decoder_output, decoder_hidden = self.model.decoder(
                        decoder_input, decoder_hidden, enc_out)
                    log_prob, indices = decoder_output.data.topk(self.beam_size)
                    for k in range(self.beam_size):
                        index = indices[:,k].unsqueeze(1).detach()
                        log_p = log_prob[0,k].item()
                        child = Node(decoder_hidden, node, index, node.log_prob + log_p, node.length + 1)
                        candidates.append((node.log_prob + log_p, child))
                # Sort candidates by prob
                candidates = sorted(candidates, key=lambda x:x[0], reverse=True)
                # Select top K, select all if not enough
                length = min(len(candidates), self.beam_size)
                for i in range(length):
                    q.put(candidates[i][1])

            candidates = []
            for node in end_nodes:
                value = node.log_prob
                candidates.append((value, node))
            
            candidates = sorted(candidates, key=lambda x:x[0], reverse=True)
            
            node = candidates[0][1]
            while node.previous_node != None:
                tgt_id.append(node.decoder_input.item())
                node = node.previous_node
            tgt_id = tgt_id[::-1][:-1]
            tgt_id = ' '.join(map(str, tgt_id))
            return tgt_id
    
    def translate_v(self, src_id, src_vfeat):
        logLikelihoods = []
        preds = []
        atten_probs = []
        coverage_penalties = []
        beam_size = self.beam_size
        remaining_beams = self.beam_size
        BOS_id = 2
        EOS_id = 3
        
        # generate context from source
#         src_id = self.word2id(source)
#         src_pieces = [self.id2word(str(i)) for i in src_id]
        with torch.no_grad():
            src_id = Variable(torch.LongTensor(src_id).unsqueeze(0).cuda(), volatile=True)
            src_vfeat = Variable(torch.tensor(src_vfeat).float().cuda(), volatile=True)
            input_length = src_id.shape[1]
            enc_out_t, hidden = self.model.encoder_t(src_id)
            enc_out_v, hidden_v = self.model.encoder_v(src_vfeat)

            # predict the first word
            decoder_input = Variable(torch.LongTensor([BOS_id]).unsqueeze(0).cuda())
            out, hidden = self.model.decoder(decoder_input, hidden, enc_out_t, enc_out_v)
            scores, scores_id = out.view(-1).topk(beam_size)
            node = Node(hidden, None, decoder_input, 0, 1)
            q = Queue()
            q.put(node)
            end_nodes = []
            tgt_id = []
            while not q.empty():
                candidates = []
                # level traversal
                for _ in range(q.qsize()):
                    node = q.get()
                    decoder_input = node.decoder_input
                    decoder_hidden = node.hidden
                    if decoder_input.item() == EOS_id or node.length >= self.max_len:
                        end_nodes.append(node)
                        continue
                    decoder_output, decoder_hidden = self.model.decoder(
                        decoder_input, decoder_hidden, enc_out_t, enc_out_v)
                    log_prob, indices = decoder_output.data.topk(self.beam_size)
                    for k in range(self.beam_size):
                        index = indices[:,k].unsqueeze(1).detach()
                        log_p = log_prob[0,k].item()
                        child = Node(decoder_hidden, node, index, node.log_prob + log_p, node.length + 1)
                        candidates.append((node.log_prob + log_p, child))
                # Sort candidates by prob
                candidates = sorted(candidates, key=lambda x:x[0], reverse=True)
                # Select top K, select all if not enough
                length = min(len(candidates), self.beam_size)
                for i in range(length):
                    q.put(candidates[i][1])

            candidates = []
            for node in end_nodes:
                value = node.log_prob
                candidates.append((value, node))
            
            candidates = sorted(candidates, key=lambda x:x[0], reverse=True)
            
            node = candidates[0][1]
            while node.previous_node != None:
                tgt_id.append(node.decoder_input.item())
                node = node.previous_node
            tgt_id = tgt_id[::-1][:-1]
            tgt_id = ' '.join(map(str, tgt_id))
            return tgt_id