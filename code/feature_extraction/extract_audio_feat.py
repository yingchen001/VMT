import os
import torch
import subprocess
import numpy as np
from tqdm import tqdm
from glob import glob
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='3'

# Load VGGish model
model = torch.hub.load('harritaylor/torchvggish', 'vggish')
# model = model.to('cuda')
model.eval()

# Load training and validation video lists
train_list = glob('/mnt/md0/yingchen_ntu/VMT/VMT/VATEX/train/*.mp4')
val_list = glob('/mnt/md0/yingchen_ntu/VMT/VMT/VATEX/valid/*.mp4')
sorted(train_list)
sorted(val_list)
discarded = []
for f_list in [train_list, val_list]:
    audio_dir = '/mnt/md0/yingchen_ntu/VMT/VMT/yc_video_transformer/data/audios'
    audio_feat_dir = '/mnt/md0/yingchen_ntu/VMT/VMT/yc_video_transformer/data/audio_features'
    for path in tqdm(f_list):
        try:
            vid = path.split('/')[-1].replace('.mp4','')
            audio_path = os.path.join(audio_dir, vid+'.wav')
            audio_feat_path = os.path.join(audio_feat_dir, vid+'.npy')
            ## Convert video to wav, suggest to do it in a seperate script
            # command = "ffmpeg -i {} -ab 160k -ac 2 -ar 16000 -vn {}".format(path, audio_path)
            # subprocess.call(command, shell=True)
            with torch.no_grad():
                pred = model.forward(audio_path)
            np.save(audio_feat_path, pred.detach().numpy().astype(np.float32))
        except Exception as e:
            print(e)
            discarded.append(path)
print('Following videos are discarded due to poor quality or no corresponding audio found')
print(discarded)