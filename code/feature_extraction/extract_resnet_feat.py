import os
import torch
import numpy as np
from tqdm import tqdm
from glob import glob
import torchvision
from torchvision import transforms

# Load pretrain resnext-101 model, remove last layer to extract features
model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')
model.eval()
model2 = torch.nn.Sequential(*(list(model.children())[:-1])).cuda()
model2.eval()

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# Load video lists
train_list = glob('/mnt/md0/yingchen_ntu/VMT/VMT/VATEX/train/*.mp4')
val_list = glob('/mnt/md0/yingchen_ntu/VMT/VMT/VATEX/valid/*.mp4')
sorted(train_list)
sorted(val_list)
discarded = []
for f_list in [train_list, val_list]:
    img_feat_dir = '/mnt/md0/yingchen_ntu/VMT/VMT/yc_video_transformer/data/resnet_features'
    for path in tqdm(f_list):
        try:
            vid = path.split('/')[-1].replace('.mp4','')
            i3d_feat_path = os.path.join('/mnt/md0/yingchen_ntu/VMT/VMT/yc_video_transformer/data/vatex_features/trainval', vid+'.npy')
            feat_path1 = os.path.join(img_feat_dir, 'matched_size', vid+'.npy')
            feat_path2 = os.path.join(img_feat_dir, 'size32', vid+'.npy')
            feat_size = np.load(i3d_feat_path).shape[1]
            video = torchvision.io.read_video(path)[0]
            video_len = video.shape[0]
            # Transform the video frames to ?x3x224x224 for model prediction
            batch1 = video[np.linspace(0,video_len-1,feat_size).astype(int)] # match i3d size
            batch2 = video[np.linspace(0,video_len-1,32).astype(int)] # 32
            batch1 = torch.stack([preprocess(video[i].permute(2,0,1)) for i in range(batch1.shape[0])]).cuda()
            batch2 = torch.stack([preprocess(video[i].permute(2,0,1)) for i in range(batch2.shape[0])]).cuda()
            with torch.no_grad():
                feat1 = model2(batch1)
                feat2 = model2(batch2)
            np.save(feat_path1, feat1.cpu().squeeze().detach().numpy().astype(np.float32))
            np.save(feat_path2, feat2.cpu().squeeze().detach().numpy().astype(np.float32))
        except Exception as e:
            print(e)
            discarded.append(path)
print('Following videos are discarded due to poor quality')
print(discarded)