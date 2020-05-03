
# Multi-modal Information Fusion for Video-guided Machine Translation
This repository is for [CE7455](https://ntunlpsg.github.io/ce7455_deep-nlp-20/) Project: Multi-modal Information Fusion for Video-guided Machine Translation introduced in the report. 
This code is built on [Video-guided-Machine-Translation](https://github.com/eric-xw/Video-guided-Machine-Translation) and [Transformer-py](https://github.com/zhangxiangnick/Transformer-py), two VMT models are implemented in this repo, which are Transformer-based VMT model and Multi-modal Video Representations S2S model respectively.
## Prerequisites 
- Python 3.7
- PyTorch 1.4 (1.0+)
- nltk 3.4.5
- sacrebleu
- numpy 
- pandas
- tqdm
- torchvision

## Training & Evaluation

### 1. Download VATEX dataset, including corpus files, videos, and extracted i3D features
Details can be found at https://vatex.org/main/download.html, please change corresponding paths for above data after downloading.
### 2. Feature Extraction
Run ```./feature_extraction/extract_resnet_feat.py``` and ```./feature_extraction/extract_audio_feat.py``` to extract the apperance and audio features from raw videos respectively.
### 3. Training & Validation
#### Baseline (https://arxiv.org/abs/1904.03493)
To train and evaluate the baseline NMT model (w/o i3D features):
```
python train_baseline_nmt.py
```
To train and evaluate the baseline VMT model (w/ i3D features):
```
python train_baseline.py
```
The default hyperparamters are set in `configs/configs.yaml`. 

#### Multi-modal Video Representations S2S Model
To train and evaluate the Multi-modal LSTM model:
```
python train_multi.py
```
The default hyperparamters are set in `configs/configs_multi.yaml`. 

#### Transformer-based VMT Model
To train and evaluate the transformer model (w/o i3D features):
```
python train_transformer.py
```
To train and evaluate the transformer VMT model (w/ i3D features):
```
python train_vtransformer.py
```
The default hyperparamters are set in `configs/configs_transformer.yaml`. 

## Acknowledge
This code is built on [Video-guided-Machine-Translation](https://github.com/eric-xw/Video-guided-Machine-Translation) and [Transformer-py](https://github.com/zhangxiangnick/Transformer-py). We thank the authors for sharing their codes.
