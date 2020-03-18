# Install

Follow `fairseq` installation, then:

````
# Chinese tokenizer
$ pip install jieba

# English tokenizer
$ pip install nltk
$ mkdir -p ~/nltk_data/tokenizers/
$ wget https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt.zip -o ~/nltk_data/tokenizers/punkt.zip
$ unzip ~/nltk_data/tokenizers/punkt.zip ~/nltk_data/tokenizers/

````

Additionally, we use scripts from Moses and Subword-nmt

````
git clone https://github.com/moses-smt/mosesdecoder
````

````
git clone https://github.com/rsennrich/subword-nmt
````
