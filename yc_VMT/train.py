import os
import math
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from Model import Transformer
from Dataloader import Dataloader
from Optimizer import TransformerOptimizer
from tqdm import tqdm
import pickle
from Translator import Translator
import sacrebleu

def trainEpoch(epoch, model, criterion, dataloader, optim, print_batch=100):
    model.train()
    epoch_loss, epoch_words, epoch_corrects = 0, 0, 0
    batch_loss, batch_words, batch_corrects = 0, 0, 0
    batch_size = dataloader.batch_size
    start = time.time()
    for i in range(len(dataloader)):
        src_batch, tgt_batch = dataloader[i]
        model.zero_grad()
        # leave out the last <EOS> in target
        out, _ = model(src_batch, tgt_batch[:,:-1])  
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
        for i in range(len(dataloader)):
            src_batch, tgt_batch = dataloader[i]
            out, _ = model(src_batch, tgt_batch[:, :-1])
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
def index2sentence(input, lang='english'):
    if isinstance(input, list):
        indexes = [int(x) for x in input]
    elif isinstance(input, str):
        indexes = [int(x) for x in input.split()]
    else:
        print('not suppported input', input)
    # filter indexes and remove 1
    indexes = [i for i in indexes if i!=1]
    if lang == 'english':
        words = [english_dict['index2word'][index] for index in indexes]
    elif lang == 'chinese':
        words = [chinese_dict['index2word'][index] for index in indexes]
    return ' '.join(words)

def test(model, dataloader, target_lang='chinese', max_steps=None, verbose=True):
    dataloader.shuffle(1024)
    translator = Translator(model, 'no-bpe')
    preds, refs = [], []
    if target_lang=='chinese':
        source_lang = 'english'
    else:
        source_lang = 'chinese'
    for i, (src, tgt) in tqdm(enumerate(dataloader)):
        if max_steps and i >= max_steps:
            break
        x = src.tolist()[0]
        y = tgt.tolist()[0]
        try:
            pred = translator.translate(x)
            src_sentence = index2sentence(x, source_lang)
            tgt_sentence = index2sentence(y, target_lang)
            pred_sentence = index2sentence(pred, target_lang)
            preds.append(pred_sentence)
            refs.append(tgt_sentence)
            if verbose:
                print()
                print('src', src_sentence)
                print('tgt', tgt_sentence)
                print('pred', pred_sentence)
        except Exception as e:
            print('failed to predict due to error:',e)
    bleu = sacrebleu.corpus_bleu(preds, [refs])
    return bleu

if __name__ == "__main__":
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    # params
    num_epochs = 100
    batch_size = 128
    run_testing_during_training = True

    # preprocessing_type = 'singleword'
    # preprocessing_type = 'new'
    preprocessing_type = 'single_plus'

    print('preprocessing: ', preprocessing_type)

    vocab_size = 30000
    if preprocessing_type == 'single_plus':
        vocab_size = 68000

    # load the dictionary for testing
    if run_testing_during_training:
        chinese_dict = pickle.load(open('./data_'+preprocessing_type +'/chinese_dict.pkl', 'rb'))
        english_dict = pickle.load(open('./data_'+preprocessing_type +'/english_dict.pkl', 'rb'))
        print('chineses and english dictionaries loaded..')

    print("Building Dataloader ...")
    train_path1 = './data_'+preprocessing_type +'/train.clean.en.id'
    train_path2 = './data_'+preprocessing_type +'/train.clean.zh.id'
    valid_path1 = './data_'+preprocessing_type +'/valid.clean.en.id'
    valid_path2 = './data_'+preprocessing_type +'/valid.clean.zh.id'
    test_path1 = './data_'+preprocessing_type +'/test.clean.en.id'
    test_path2 = './data_'+preprocessing_type +'/test.clean.zh.id'

    traindataloader = Dataloader(train_path1, train_path2, batch_size, cuda=True)
    devdataloader = Dataloader(valid_path1, valid_path2, batch_size, cuda=True, volatile=True)
    if run_testing_during_training:  
        testdataloader = Dataloader(test_path1, test_path2, 1, cuda=True, volatile=True)  # test sentences one by one
    
    print("Building Model ...")
    model = Transformer(bpe_size=vocab_size, h=8, d_model=512, p=0.1, d_ff=1024).cuda()
    nllloss_weights = torch.ones(vocab_size)  
    criterion = nn.NLLLoss(nllloss_weights, size_average=False, ignore_index=0).cuda()
    # criterion = nn.NLLLoss(size_average=False, ignore_index=0).cuda()
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

        # testing every 10 epochs
        if (epoch % 10 == 0) or epoch == num_epochs - 1:
            print('running testing...')
            try:
                bleu = test(model, testdataloader, target_lang='chinese', max_steps=100, verbose=False)
                print('BLEU score at epoch {} : {}'.format(epoch, bleu))
            except Exception as e:
                print('unable to get bleu due to: ', e)
