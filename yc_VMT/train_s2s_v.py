import os
import math
import time
import torch
import pickle
import sacrebleu
from tqdm import tqdm
import torch.nn as nn
from torch.autograd import Variable
from preprocess import Lang
from Layers import EncoderRNN, EncoderRNN_VFeat, AttnDecoderRNN_V, \
                    Attention, AttnDecoderRNN, AttnDecoderRNN_V
from Model import Transformer, Seq2Seq, Seq2Seq_VFeat
from Dataloader import Dataloader
from Optimizer import TransformerOptimizer
from Translator_s2s import Translator

BOS_id = 2
EOS_id = 3

def trainEpoch(epoch, model, criterion, dataloader, optim, print_batch=100):
    model.train()
    epoch_loss, epoch_words, epoch_corrects = 0, 0, 0
    batch_loss, batch_words, batch_corrects = 0, 0, 0
    batch_size = dataloader.batch_size
    start = time.time()
    for i in range(len(dataloader)):
        src_batch, tgt_batch = dataloader[i]
        src_batch_t, src_batch_v = src_batch[0], src_batch[1]
        model.zero_grad()
        # leave out the last <EOS> in target
        out = model(src_batch_t, src_batch_v, tgt_batch[:,:-1])
        out = out.contiguous().view(-1, model.decoder.output_dim) 
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
            
    
    # epoch_accuracy = epoch_corrects/epoch_words
    # epoch_perplexity = math.exp(epoch_loss*batch_size/epoch_words)
    # return epoch_accuracy, epoch_perplexity
    return 0, 0

def evaluate(epoch, model, criterion, dataloader):
    model.eval()
    eval_loss, eval_words, eval_corrects = 0, 0, 0
    with torch.no_grad():
        for i in range(len(dataloader)):
            src_batch, tgt_batch = dataloader[i]
            src_batch_t, src_batch_v = src_batch[0], src_batch[1]
            out = model(src_batch_t, src_batch_v, tgt_batch[:, :-1])
            out = out.contiguous().view(-1, model.decoder.output_dim) 
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

# for testing during training
def index2sentence(input, dict):
    if isinstance(input, list):
        indexes = [int(x) for x in input]
    elif isinstance(input, str):
        indexes = [int(x) for x in input.split()]
    else:
        print('not suppported input', input)
    # filter indexes and remove BOS_id
    indexes = [i for i in indexes if i!=BOS_id]
    words = [dict.index2word[index] for index in indexes]
    return ' '.join(words)

def test(model, dataloader, src_lang, tgt_lang, src_dict, tgt_dict, max_steps=None, verbose=True):
    translator = Translator(model, max_len=40, beam_size=5)
    preds, refs = [], []
    with open('./data/test.{}'.format(tgt_lang), 'r') as f:
        refs = f.readlines()
    for i, (src, tgt) in tqdm(enumerate(dataloader)):
        if max_steps and i >= max_steps:
            break
        x_t, x_v = src[0], src[1]
        x_t = x_t.tolist()[0]
        y = tgt.tolist()[0]
        try:
            pred = translator.translate_v(x_t, x_v)
            src_sentence = index2sentence(x_t, src_dict)
            tgt_sentence = refs[i]
            pred_sentence = index2sentence(pred, tgt_dict)
            preds.append(pred_sentence)
            refs.append(tgt_sentence)
            if verbose:
                print()
                print('src', src_sentence)
                print('tgt', tgt_sentence)
                print('pred', pred_sentence)
        except Exception as e:
            print('failed to predict due to error:',e)
            break
    if tgt_lang == 'zh':
        preds = [sent.replace(' ','') for sent in preds]
        refs = [sent.replace(' ','') for sent in refs]
        bleu = sacrebleu.corpus_bleu(preds, [refs], tokenize = 'zh')
    else:
        bleu = sacrebleu.corpus_bleu(preds, [refs])
    return bleu

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    # params
    num_epochs = 100
    batch_size = 128
    MAX_LEN = 40
    src_lang = 'en'
    tgt_lang = 'zh'
    run_testing_during_training = True
    preprocessing_type = 'jieba'
    print('Loading dict')
    src_dict = pickle.load(open('./data/{}/{}_dict.pkl'.format(preprocessing_type, src_lang), 'rb'))
    tgt_dict = pickle.load(open('./data/{}/{}_dict.pkl'.format(preprocessing_type, tgt_lang), 'rb'))
    print("Building Dataloader ...")
    train_path = './data/{}/train.id'.format(preprocessing_type)
    valid_path = './data/{}/valid.id'.format(preprocessing_type)
    test_path = './data/{}/test.id'.format(preprocessing_type)

    traindataloader = Dataloader(train_path, batch_size, src_lang=src_lang, tgt_lang=tgt_lang,
                                 v_feat='i3d',max_len=MAX_LEN, cuda=True)
    devdataloader = Dataloader(valid_path, batch_size, src_lang=src_lang, tgt_lang=tgt_lang, 
                                v_feat='i3d', max_len=MAX_LEN, cuda=True, volatile=True)
    if run_testing_during_training:  
        testdataloader = Dataloader(test_path, 1, src_lang=src_lang, tgt_lang=tgt_lang, 
                                v_feat='i3d', max_len=MAX_LEN, cuda=True, volatile=True, sort=False)  # test sentences one by one
    
    print("Building Model ...")
    INPUT_DIM = src_dict.n_words + 1
    OUTPUT_DIM = tgt_dict.n_words + 1
    ENC_EMB_DIM = 512
    ENC_V_DIM = 1024
    DEC_EMB_DIM = 512
    ENC_HID_DIM = 512
    DEC_HID_DIM = 1024

    attn = Attention(ENC_HID_DIM*2, DEC_HID_DIM)
    enc_t = EncoderRNN(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM)
    enc_v = EncoderRNN_VFeat(ENC_V_DIM, ENC_HID_DIM)
    dec = AttnDecoderRNN_V(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM*2, DEC_HID_DIM, attn)

    model = Seq2Seq_VFeat(enc_t, enc_v, dec, device).to(device)
    print(model)
    print(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')
    # nllloss_weights = torch.ones(vocab_size)  
    # criterion = nn.CrossEntropyLoss(ignore_index = 0)
    criterion = nn.NLLLoss(size_average=False, ignore_index=0).cuda()
    base_optim = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09)
    optim = TransformerOptimizer(base_optim, warmup_steps=32000, d_model=512)

    print("Start Training ...")
    best_eval_acc = 0
    for epoch in range(num_epochs):
        if epoch > 0:
            traindataloader.shuffle(1024)
        if epoch == 20:
            optim.init_lr = 0.5 * optim.init_lr 
        if epoch == 40:
            optim.init_lr = 0.3 * optim.init_lr 
        if epoch == 70:
            optim.init_lr = 0.3 * optim.init_lr 
        train_acc, train_ppl= trainEpoch(epoch, model, criterion, traindataloader, optim)
        print("[Train][Epoch %2d] Accuracy: %6.2f, Perplexity: %6.2f" % (epoch+1, train_acc, train_ppl))
        eval_acc, eval_ppl = evaluate(epoch, model, criterion, devdataloader)
        print("[Eval][Epoch %2d] Accuracy: %6.2f, Perplexity: %6.2f" % (epoch+1, eval_acc, eval_ppl))
        checkpoint = {'model': model.state_dict(),
                      'epoch': epoch, 'optimizer': optim}
        checkpoint_folder = 'checkpoints/' + preprocessing_type
        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder)
        checkpoint_path = os.path.join(checkpoint_folder, 'epoch%d_acc_%.2f_ppl_%.2f.pt' % (epoch+1, 100*eval_acc, eval_ppl))
        if eval_acc > best_eval_acc:
            print('eval acc improved from {} to {}'.format(best_eval_acc, eval_acc))
            print('save checkpoint to {}'.format(checkpoint_path))
            best_eval_acc = eval_acc
            torch.save(checkpoint, checkpoint_path)
        bleu = test(model, testdataloader, src_lang, tgt_lang, src_dict, tgt_dict)
        print('BLEU score at epoch {} : {}'.format(epoch, bleu))
        # testing every 10 epochs
        if (epoch % 10 == 0) or epoch == num_epochs - 1:
            print('running testing...')
            try:
                bleu = test(model, testdataloader, src_lang, tgt_lang, src_dict, tgt_dict, max_steps=None, verbose=False)
                print('BLEU score at epoch {} : {}'.format(epoch, bleu))
            except Exception as e:
                print('unable to get bleu due to: ', e)
