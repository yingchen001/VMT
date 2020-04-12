import os
# import skvideo.io  
# videodata = skvideo.io.vread("/mnt/md0/yingchen_ntu/VMT/VMT/VATEX/valid/_2zXAtOKXm0_000004_000014.mp4")  
import json
import tqdm
import pandas as pd

def prepare_list(data_list, video_path, i3d_path):
    sent_pairs = []
    for v in tqdm.tqdm(data_list):
        vid = v['videoID']
        # assert len(v['enCap'])==len(v['chCap'])==10, 'please check {}'.format(v['videoID'])
        for i in range(5):
            sent_pair = {}
            sent_pair['id'] = vid + '_{}'.format(i)
            sent_pair['en'] = v['enCap'][i+5]
            sent_pair['zh'] = v['chCap'][i+5]
            v_match = os.path.join(video_path, vid+'.mp4')
            sent_pair['v_path'] = v_match if os.path.isfile(v_match) else ''
            i3d_match = os.path.join(i3d_path, vid+'.npy')
            sent_pair['i3d_path'] = i3d_match if os.path.isfile(i3d_match) else ''
            sent_pairs.append(sent_pair)
    return sent_pairs

if __name__ == "__main__":
    train_path = '/mnt/md0/yingchen_ntu/VMT/VMT/VATEX/vatex_training_v1.0.json'
    # Use validation set as test
    test_path = '/mnt/md0/yingchen_ntu/VMT/VMT/VATEX/vatex_validation_v1.0.json'
    i3d_path = '/mnt/md0/yingchen_ntu/VMT/VMT/VATEX/i3d/'
    video_train_path = '/mnt/md0/yingchen_ntu/VMT/VMT/VATEX/train/'
    video_valid_path = '/mnt/md0/yingchen_ntu/VMT/VMT/VATEX/valid/'
    with open(train_path) as j:
        train_vid = json.load(j)
        # Split train into train and valid
        valid_vid = train_vid[-3000:]
        train_vid = train_vid[:-3000]

    with open(test_path) as j:
        test_vid = json.load(j)

    print('preparing training\n')
    train_data = prepare_list(train_vid, video_train_path, i3d_path)
    print('preparing validation\n')
    val_data = prepare_list(valid_vid, video_train_path, i3d_path)
    print('preparing testing\n')
    test_data = prepare_list(test_vid, video_valid_path, i3d_path)

    train_data = pd.DataFrame(train_data)
    train_data.to_csv('./data/vatex_train.csv')
    val_data = pd.DataFrame(val_data)
    val_data.to_csv('./data/vatex_valid.csv')
    test_data = pd.DataFrame(test_data)
    test_data.to_csv('./data/vatex_test.csv')