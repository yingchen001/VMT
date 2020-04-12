import os
import jieba
import nltk
import pickle
import tqdm
import pandas as pd

use_jieba = True
UNK_id = 1
BOS_id = 2
EOS_id = 3
class Lang:
    def __init__(self, lang):
        self.lang = lang
        self.word2index = {}
        self.word2count = {}
        self.index2word = {1: "<unk>", 2: "<BOS>", 3: "<EOS>"}
        self.n_words = 4 

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
    
    def removeLowFreq(self, min_freq=5):
        print('Removing Low Frequency Words (min_freq={})'.format(min_freq))
        print('Original size is {}'.format(self.n_words))
        low_freq_words = [key for key, value in self.word2count.items() if value < min_freq]
        self.index2word = {key:value for key, value in self.index2word.items() if value not in low_freq_words}
        self.index2word = {i+1:value for i, (key,value) in enumerate(self.index2word.items()) if i+1 > 3}
        self.index2word.update({1: "<unk>", 2: "<BOS>", 3: "<EOS>"})
        self.word2index = {value:key for key, value in self.index2word.items()}
        self.word2count = {key:value for key, value in self.word2count.items() if key not in low_freq_words}
        self.n_words = len(self.word2index)
        print('Number of words left in {} is {}'.format(self.lang, self.n_words))

def raw2id(df, en_dict, zh_dict, use_jieba=True):
    df_id = df.copy()
    for index, row in df.iterrows():
        zh_line, en_line = row['zh'], row['en']
        if use_jieba:
            zh_id = [zh_dict.word2index[tok] if tok in zh_dict.word2index else UNK_id for tok in jieba.lcut(zh_line)]
        else:
            zh_id = [zh_dict.word2index[tok] if tok in zh_dict.word2index else UNK_id for tok in zh_line]
        zh_id = ' '.join([str(x) for x in [BOS_id]+zh_id+[EOS_id]])

        en_id = [en_dict.word2index[tok] if tok in en_dict.word2index else UNK_id for tok in nltk.word_tokenize(en_line)]
        en_id = ' '.join([str(x) for x in [BOS_id]+en_id+[EOS_id]])
        df_id.at[index,'en'] = en_id
        df_id.at[index,'zh'] = zh_id
    return df_id

if __name__ == "__main__":
    en_dict = Lang('en')
    zh_dict = Lang('zh')
    vatex_train = pd.read_csv('./data/vatex_train.csv')
    vatex_valid= pd.read_csv('./data/vatex_valid.csv')
    vatex_test = pd.read_csv('./data/vatex_test.csv')
    raw_zh = vatex_train['zh'].to_list()
    raw_en = vatex_train['en'].to_list()
    # Prepare word dict
    assert len(raw_zh) ==  len(raw_en), 'invalid data with different length'
    for i in range(len(raw_zh)):
        zh_line, en_line = raw_zh[i], raw_en[i]
        if use_jieba:
            zh_tok = ' '.join(jieba.lcut(zh_line))
        else:
            zh_tok = ' '.join([x for x in zh_line])
        en_tok = ' '.join(nltk.word_tokenize(en_line))
        en_dict.addSentence(en_tok)
        zh_dict.addSentence(zh_tok)

    en_dict.removeLowFreq()
    zh_dict.removeLowFreq()

    # Save Word Dict
    dict_root = './data/jieba' if use_jieba else './data/single'
    with open(os.path.join(dict_root, 'zh_dict.pkl'), 'wb') as f:
        pickle.dump(zh_dict, f)
    with open(os.path.join(dict_root, 'en_dict.pkl'), 'wb') as f:
        pickle.dump(en_dict, f)

    # Convert raw data to index
    train_id = raw2id(vatex_train, en_dict, zh_dict)
    valid_id = raw2id(vatex_valid, en_dict, zh_dict)
    test_id = raw2id(vatex_test, en_dict, zh_dict)
    train_id.to_csv(os.path.join(dict_root, 'train.id'), index=False)
    valid_id.to_csv(os.path.join(dict_root, 'valid.id'), index=False)
    test_id.to_csv(os.path.join(dict_root, 'test.id'), index=False)

