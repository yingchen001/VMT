import os
import math
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from model_transformer import Transformer, TransformerNew, TransformerVideo
# from Dataloader import Dataloader
from Optimizer import TransformerOptimizer
from tqdm import tqdm
import pickle
from Translator import VTranslator
import sacrebleu
import json
import numpy as np
import logging
import datetime
import argparse
import sys

from dataloader import create_split_loaders
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from utils import set_logger,read_vocab,write_vocab,build_vocab,Tokenizer,padding_idx,clip_gradient,adjust_learning_rate
cc = SmoothingFunction()

class Arguments():
    def __init__(self, config):
        for key in config:
            setattr(self, key, config[key])

def setup(args, clear=False):
    '''
    Build vocabs from train or train/val set.
    '''
    TRAIN_VOCAB_EN, TRAIN_VOCAB_ZH = args.TRAIN_VOCAB_EN, args.TRAIN_VOCAB_ZH
    if clear: ## delete previous vocab
        for file in [TRAIN_VOCAB_EN, TRAIN_VOCAB_ZH]:
            if os.path.exists(file):
                os.remove(file)
    # Build English vocabs
    if not os.path.exists(TRAIN_VOCAB_EN):
        write_vocab(build_vocab(args.DATA_DIR, language='en'),  TRAIN_VOCAB_EN)
    #build Chinese vocabs
    if not os.path.exists(TRAIN_VOCAB_ZH):
        write_vocab(build_vocab(args.DATA_DIR, language='zh'), TRAIN_VOCAB_ZH)

    # set up seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

def trainEpoch(epoch, model, criterion, dataloader, optim, print_batch=100):
    model.train()
    epoch_loss, epoch_words, epoch_corrects = 0, 0, 0
    batch_loss, batch_words, batch_corrects = 0, 0, 0
    batch_size = dataloader.batch_size
    start = time.time()
    for i, (srccap, tgtcap, video, caplen_src, caplen_tgt, srcrefs, tgtrefs) in enumerate(dataloader):
        # src_batch, tgt_batch = dataloader[i]
        # for new loader
        src_batch = srccap.cuda()
        tgt_batch = tgtcap.cuda()
        video = video.cuda()
        # print('input', src_batch.shape, tgt_batch.shape, video.shape)
        model.zero_grad()
        # leave out the last <EOS> in target
        # out, _ = model(src_batch, tgt_batch[:,:], video)  
        out, _ = model(src_batch, tgt_batch[:,:-1], video)
        # label smoothing 
        # randomly set 10% target labels to 0, which doesn't contribute to loss
        labelsmoothing_mask = torch.le(torch.zeros(tgt_batch[:,1:].size()).uniform_(), 0.1).cuda()
        tgt_words = tgt_batch[:,1:].data.clone().masked_fill_(labelsmoothing_mask, 0)
        tgt_words = Variable(tgt_words.contiguous().view(-1))    
        loss = criterion(out, tgt_words) / batch_size   
        loss.backward()
        optim.step()
        preds = torch.max(out,1)[1]        
        corrects = preds.data.eq(tgt_words.data).masked_select(tgt_words.data.ne(0))          
        batch_loss += loss.item()     
        batch_words += len(corrects)      
        batch_corrects += corrects.sum().item()
        if (i+1)%print_batch==0 or (i+1)==len(dataloader):
            print("Epoch %2d, Batch %6d/%6d, Acc: %6.2f, Plp: %8.2f, %4.0f seconds" % 
                 (epoch+1, i+1, len(dataloader), batch_corrects/batch_words, 
                  math.exp(batch_loss*batch_size/batch_words), time.time()-start))
            epoch_loss += batch_loss
            epoch_words += batch_words
            epoch_corrects += batch_corrects
            batch_loss, batch_words, batch_corrects = 0, 0, 0
            start = time.time()
    epoch_accuracy = epoch_corrects/epoch_words
    epoch_perplexity = math.exp(epoch_loss*batch_size/epoch_words)
    return epoch_accuracy, epoch_perplexity

def evaluate(epoch, model, criterion, dataloader):
    model.eval()
    eval_loss, eval_words, eval_corrects = 0, 0, 0
    with torch.no_grad():
        # for i in range(len(dataloader)):
        for i, (srccap, tgtcap, video, caplen_src, caplen_tgt, srcrefs, tgtrefs) in enumerate(dataloader):
            # src_batch, tgt_batch = dataloader[i]
            src_batch = srccap.cuda()
            tgt_batch = tgtcap.cuda()
            video = video.cuda()
            
            out, _ = model(src_batch, tgt_batch[:,:-1], video)
            tgt_words = tgt_batch[:,1:].contiguous().view(-1)      
            loss = criterion(out, tgt_words)    
            preds = torch.max(out,1)[1]        
            corrects = preds.data.eq(tgt_words.data).masked_select(tgt_words.data.ne(0))          
            eval_loss += loss.item()     
            eval_words += len(corrects)      
            eval_corrects += corrects.sum().item()
        eval_accuracy = eval_corrects/eval_words
        eval_perplexity = math.exp(eval_loss/eval_words)
    return eval_accuracy, eval_perplexity

# def test(model, dataloader, src_tokenizor, tgt_tokenizor, max_steps=None, print_every=-1):
#     '''
#         max_steps: run how many steps for each test
#     '''
#     # load translator
#     translator = Translator(model, 'no-bpe')

#     preds, refs = [], []
#     for i, (srccap, tgtcap, video, caplen_src, caplen_tgt, srcrefs, tgtrefs) in enumerate(dataloader):
#         if max_steps and i >= max_steps:
#             break
#         x = srccap.tolist()[0]
#         y = tgtcap.tolist()[0]
#         try:
#             pred = translator.translate(x) # pred is a str by joining indexes
#             pred_list = map(int, pred.split(' '))
#             src_sentence = src_tokenizor.decode_sentence(x)
#             tgt_sentence = tgt_tokenizor.decode_sentence(y)
#             pred_sentence = tgt_tokenizor.decode_sentence(pred_list)
#             preds.append(pred_sentence)
#             refs.append(tgt_sentence)
#             if print_every > 0 and i % print_every == 0:
#                 print()
#                 print('src', src_sentence)
#                 print('tgt', tgt_sentence)
#                 print('pred', pred_sentence)
#         except Exception as e:
#             # print('failed to predict due to error:',e)
#             pass
#     bleu = sacrebleu.corpus_bleu(preds, [refs])
#     return bleu

def test_nltk(model, dataloader, src_tokenizor, tgt_tokenizor, max_steps=None, print_every=-1):
    '''
        max_steps: run how many steps for each test
    '''
    # load translator
    translator = VTranslator(model, 'no-bpe')

    hypotheses, refs = [], []
    for i, (srccap, tgtcap, video, caplen_src, caplen_tgt, srcrefs, tgtrefs) in enumerate(tqdm(dataloader)):
        if max_steps and i > max_steps:
            break
        x = srccap.tolist()[0]
        y = tgtcap.tolist()[0]
        video = video.cuda()
        ref = tgtrefs[0]
        ref = [list(map(int, ref.split()))]
        try:
            pred = translator.translate(x, video) # pred is a str by joining indexes
            pred_list = list(map(int, pred.split(' ')))
            hypo = pred_list[1:-1]
            hypotheses.append(hypo)
            refs.append(ref)
            src_sentence = src_tokenizor.decode_sentence(x[1:], filter_EOS=True)
            tgt_sentence = tgt_tokenizor.decode_sentence(y[1:], filter_EOS=True)
            pred_sentence = tgt_tokenizor.decode_sentence(pred_list[1:], filter_EOS=True)
            if print_every > 0 and i % print_every == 0:
                print()
                print('src', src_sentence)
                print('tgt', tgt_sentence)
                print('pred', pred_sentence)
        except Exception as e:
            print('failed to predict due to error:',e)
            # pass
    corpbleu = corpus_bleu(refs, hypotheses)
    sentbleu = 0
    for i, (r, h) in enumerate(zip(refs, hypotheses), 1):
        sentbleu += sentence_bleu(r, h, smoothing_function=cc.method7)
    sentbleu /= i
    # print('\ncorpus bleu, sent bleu', corpbleu, sentbleu)
    return corpbleu, sentbleu

def get_dataloaders(args):
    model_prefix = '{}_{}'.format(args.model_type, args.train_id)
    
    log_path = args.LOG_DIR + model_prefix + '/'
    checkpoint_path = args.CHK_DIR + model_prefix + '/'
    result_path = args.RESULT_DIR + model_prefix + '/'
    cp_file = checkpoint_path + "best_model.pth.tar"
    init_epoch = 0

    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    ## set up the logger
    set_logger(os.path.join(log_path, 'train.log'))

    ## save argparse parameters
    with open(log_path+'args.yaml', 'w') as f:
        for k, v in args.__dict__.items():
            f.write('{}: {}\n'.format(k, v))

    logging.info('Training model: {}'.format(model_prefix))

    ## set up vocab txt
    # create txt here
    print('running setup')
    setup(args, clear=True)
    print(args.__dict__)

    # indicate src and tgt language
    src, tgt = 'en', 'zh'

    maps = {'en':args.TRAIN_VOCAB_EN, 'zh':args.TRAIN_VOCAB_ZH}
    vocab_src = read_vocab(maps[src])
    tok_src = Tokenizer(language=src, vocab=vocab_src, encoding_length=args.MAX_INPUT_LENGTH)
    vocab_tgt = read_vocab(maps[tgt])
    tok_tgt = Tokenizer(language=tgt, vocab=vocab_tgt, encoding_length=args.MAX_INPUT_LENGTH)
    logging.info('Vocab size src/tgt:{}/{}'.format( len(vocab_src), len(vocab_tgt)) )

    ## Setup the training, validation, and testing dataloaders
    train_loader, val_loader, test_loader = create_split_loaders(args.DATA_DIR, (tok_src, tok_tgt), args.batch_size, args.MAX_VID_LENGTH, (src, tgt), num_workers=4, pin_memory=True)
    logging.info('train/val/test size: {}/{}/{}'.format( len(train_loader), len(val_loader), len(test_loader) ))

    return train_loader, val_loader, test_loader, tok_src, tok_tgt, len(vocab_src), len(vocab_tgt)

if __name__ == "__main__":
    # arguments from configs.yaml
    parser = argparse.ArgumentParser(description='VMT')
    parser.add_argument('--config', type=str, default='./configs.yaml')
    args = parser.parse_args()
    with open(args.config, 'r') as fin:
        import yaml
        args = Arguments(yaml.load(fin))
    
    train_loader, val_loader, test_loader, tok_src, tok_tgt, src_vocab_size, tgt_vocab_size = get_dataloaders(args)
    print('data loaders loaded!')

    run_type = 'transformer_video' # for checkpoint saving
    num_epochs = 100
    num_layer = 3
    test_every = 5 # test every x epochs
    run_testing_during_training = False
    checkpoint_file = None
    # checkpoint_file = './checkpoints/transformer_novideo/epoch4_acc_48.13_ppl_17.20.pt'

    print("Building Model ...")
    # model = Transformer(bpe_size=vocab_size, h=8, d_model=512, p=0.1, d_ff=1024).cuda()
    model = TransformerVideo(src_vocab_size, tgt_vocab_size, h=8, d_model=512, p=0.1, d_ff=1024, num_layer=num_layer).cuda()
    if checkpoint_file:
        ckpt = torch.load(checkpoint_file)
        model.load_state_dict(ckpt['model'])
        start_epoch = ckpt['epoch']
        print('model loaded from checkpoint, from epoch {}'.format(start_epoch))
    else:
        start_epoch = 0

    nllloss_weights = torch.ones(tgt_vocab_size)  
    criterion = nn.NLLLoss(nllloss_weights, size_average=False, ignore_index=0).cuda()
    # criterion = nn.NLLLoss(size_average=False, ignore_index=0).cuda()
    base_optim = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09)
    optim = TransformerOptimizer(base_optim, warmup_steps=32000, d_model=512)

    print("Start Training ...")
    best_eval_acc = 0
    for epoch in range(start_epoch, num_epochs):
        if epoch > 0:
            # traindataloader.shuffle(1024)
            pass
        if epoch == 20:
            optim.init_lr = 0.5 * optim.init_lr 
        if epoch == 40:
            optim.init_lr = 0.3 * optim.init_lr 
        if epoch == 70:
            optim.init_lr = 0.3 * optim.init_lr 
        train_acc, train_ppl= trainEpoch(epoch, model, criterion, train_loader, optim)
        print("[Train][Epoch %2d] Accuracy: %6.2f, Perplexity: %6.2f" % (epoch+1, train_acc, train_ppl))
        eval_acc, eval_ppl = evaluate(epoch, model, criterion, val_loader)
        print("[Eval][Epoch %2d] Accuracy: %6.2f, Perplexity: %6.2f" % (epoch+1, eval_acc, eval_ppl))

        checkpoint = {'model': model.state_dict(),
                      'epoch': epoch, 'optimizer': optim}
        checkpoint_folder = 'checkpoints/' + run_type
        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder)
        checkpoint_path = os.path.join(checkpoint_folder, 'epoch%d_acc_%.2f_ppl_%.2f.pt' % (epoch+1, 100*eval_acc, eval_ppl))
        if eval_acc > best_eval_acc:
            print('eval acc improved from {} to {}'.format(best_eval_acc, eval_acc))
            print('save checkpoint to {}'.format(checkpoint_path))
            best_eval_acc = eval_acc
            torch.save(checkpoint, checkpoint_path)

        if (epoch % test_every == 0 and epoch > 0):
            print('running testing...')
            try:
                # bleu = test(model, test_loader, tok_src, tok_tgt, max_steps=1000, print_every=800)
                corpbleu, sentbleu = test_nltk(model, test_loader, tok_src, tok_tgt, max_steps=500, print_every=200)
                print('BLEU score at epoch {} : {}, {}'.format(epoch, corpbleu, sentbleu))
            except Exception as e:
                print('unable to get bleu due to: ', e)
        elif epoch == num_epochs - 1:
            print('running final testing...')
            try:
                corpbleu, sentbleu = test_nltk(model, test_loader, tok_src, tok_tgt, max_steps=None, print_every=200)
                print('BLEU score at epoch {} : {}, {}'.format(epoch, corpbleu, sentbleu))
            except Exception as e:
                print('unable to get bleu due to: ', e)
