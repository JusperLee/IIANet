import os

# def get_wav_files_dict(directory):
#     # 初始化空字典
#     files_dict = {}
    
#     # 遍历指定目录下的所有文件
#     for root, dirs, files in os.walk(directory):
#         for file in files:
#             # 只处理.wav文件
#             if file.endswith(".wav"):
#                 # 获取文件的绝对路径
#                 file_path = os.path.join(root, file)
#                 # 根据"_"分割文件名
#                 key = file.split('_')[0]
#                 # 如果key不在字典中，初始化一个空list
#                 if key not in files_dict:
#                     files_dict[key] = []
#                 # 将文件路径加入到对应的key的list中
#                 files_dict[key].append(file_path)
    
#     return files_dict

# # 示例用法
# directory = "/home/likai/ssd/vox2/vox2/audio_10w/wav16k/min/tr/s1"  # 替换为你的文件夹路径
# wav_files_dict = get_wav_files_dict(directory)

# # Save the dictionary to json
# import json
# with open('wav_files_dict.json', 'w') as f:
#     json.dump(wav_files_dict, f)

# Load the dictionary from json
import json
import random
import torchaudio
import torch
import numpy as np
import shutil
import yaml

from look2hear.models import IIANet
from look2hear.datas.transform import get_preprocessing_pipelines
from look2hear.videomodels import ResNetVideoModel

with open('wav_files_dict.json', 'r') as f:
    wav_files_dict = json.load(f)
    
# print(wav_files_dict)
datas = []
select_keys = random.sample(wav_files_dict.keys(), k=3)
datapath = random.sample(wav_files_dict[select_keys[0]], k=1)
mouthpath = datapath[0].split('/')[-1].split("_")
mouthpath = f"{mouthpath[0]}_{mouthpath[1]}_{mouthpath[2]}.npz"
audio_gt = torchaudio.load(datapath[0])[0]
# mouth = torch.from_numpy(np.load(mouthpath)['data'])
datas.append(audio_gt)

for key in select_keys[1:]:
    datapath = random.sample(wav_files_dict[key], k=1)
    audio = torchaudio.load(datapath[0])[0]
    sirs = torch.Tensor(1).uniform_(-30,-10).numpy()
    audio *= 10.**(sirs/20.)
    datas.append(audio)

mix = torch.stack(datas).sum(0)
torchaudio.save("mix.wav", mix, 16000)
torchaudio.save("audio_gt.wav", audio_gt, 16000)
shutil.copy(f"/home/likai/ssd/vox2/vox2/mouths/{mouthpath}", "mouth.npz")
# Load training config
with open("checkpoints/vox2/conf.yml", "rb") as f:
    train_conf = yaml.safe_load(f)

# Load model
# print(["main_args"]["exp_dir"])
checkpoint_path = os.path.join(train_conf["main_args"]["exp_dir"], "best_model.pth")
audiomodel = IIANet.from_pretrain(checkpoint_path, sample_rate=train_conf["datamodule"]["data_config"]["sample_rate"], **train_conf["audionet"]["audionet_config"])
videomodel = ResNetVideoModel(**train_conf["videonet"]["videonet_config"])
audiomodel.cuda()
audiomodel.eval()
videomodel.cuda()
videomodel.eval()

with torch.no_grad():
    mouth_roi = np.load("mouth.npz")["data"]
    mouth_roi = get_preprocessing_pipelines()["val"](mouth_roi)

    mix = torchaudio.load("mix.wav")[0].cuda()

    mouth_emb = videomodel(torch.from_numpy(mouth_roi[None, None]).float().cuda())
    est_sources = audiomodel(mix[None], mouth_emb)

    torchaudio.save("est.wav", est_sources[0].cpu(), 16000)