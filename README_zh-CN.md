简体中文 | [English](README.md)

# <font color=E7595C>I</font><font color=F6C446>I</font><font color=00C7EE>A</font><font color=00D465>Net</font>: 一种用于音视频语音分离的<font color=E7595C>内</font>部和<font color=F6C446>跨</font>模态<font color=00C7EE>注意力网络</font>

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/scanet-a-self-and-cross-attention-network-for/speech-separation-on-lrs2)](https://paperswithcode.com/sota/speech-separation-on-lrs2?p=scanet-a-self-and-cross-attention-network-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/scanet-a-self-and-cross-attention-network-for/speech-separation-on-lrs3)](https://paperswithcode.com/sota/speech-separation-on-lrs3?p=scanet-a-self-and-cross-attention-network-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/scanet-a-self-and-cross-attention-network-for/speech-separation-on-voxceleb2)](https://paperswithcode.com/sota/speech-separation-on-voxceleb2?p=scanet-a-self-and-cross-attention-network-for)
[![arXiv](https://img.shields.io/badge/arXiv-2308.08143-b31b1b.svg)](https://arxiv.org/abs/2308.08143)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://makeapullrequest.com) 
[![GitHub license](https://img.shields.io/github/license/JusperLee/IIANet.svg?color=blue)](https://github.com/JusperLee/IIANet/blob/master/LICENSE)
![GitHub stars](https://img.shields.io/github/stars/JusperLee/IIANet)
![GitHub forks](https://img.shields.io/github/forks/JusperLee/IIANet)
![Website](https://img.shields.io/website?url=https%3A%2F%2Fcslikai.cn%2FIIANet%2F&up_message=Demo%20Page&down_message=Demo%20Page&logo=webmin)


由[1]清华大学，[2]中国脑科学研究所提供。
* [李凯](https://cslikai.cn)[1]，杨润轩[1]，[孙富春](https://scholar.google.com/citations?user=DbviELoAAAAJ&hl=en)[1]，[胡晓琳](https://www.xlhu.cn/)[1,2]。

此仓库是被接受为**ICML 2024**（**海报**）的IIANet的官方实现。

## ✨主要亮点:

1. 我们提出了一种基于注意力的跨模态语音分离网络IIANet，它广泛使用了语音和视频模态内及模态间的注意力机制（IntraA和InterA）。

2. 与现有的CNN和Transformer方法相比，IIANet在三个音视频语音分离数据集上显著提高了分离质量，同时大大减少了计算复杂度和内存使用。

3. 更快的版本IIANet-fast在具有挑战性的LRS2数据集上超过了CTCNet 1.1 dB，并且仅使用了CTCNet的11%的MACs。

4. 在真实世界的YouTube场景中的定性评估显示，IIANet生成的分离语音比其他分离模型具有更高的质量。

## 🚀整体流程

<video width="900" src="https://cslikai.cn/IIANet/figures/overall.mp4" type="video/mp4">
              </video>

## 🪢IIANet架构

<video width="900" src="https://cslikai.cn/IIANet/figures/separation.mp4" type="video/mp4">
              </video>

## 🔧安装

1. 克隆仓库：

```shell
git clone https://github.com/JusperLee/IIANet.git 
cd IIANet/
```

2. 创建并激活conda环境：

```shell
conda create -n iianet python=3.8 
conda activate iianet
```

3. 按照[官方说明](https://pytorch.org)安装PyTorch和torchvision。代码要求`python>=3.8`，`pytorch>=1.11`，`torchvision>=0.13`。

4. 安装其他依赖：

```shell 
pip install -r requirements.txt
```

## 📊模型性能  

我们在三个数据集：LRS2、LRS3和VoxCeleb2上评估了IIANet及其快速版本IIANet-fast。结果显示，IIANet在保持高效率的同时，比现有方法取得了显著更好的语音分离质量 [1]。

| 方法 | 数据集 | SI-SNRi | SDRi | PESQ | 参数 | MACs | GPU推理时间 | 下载 |
|:---:|:-----:|:------:|:----:|:----:|:------:|:-----:|:-----------:|:----:|  
| IIANet | LRS2 | 16.0 | 16.2 | 3.23 | 3.1 | 18.6 | 110.11 ms | [配置](configs/LRS2-IIANet.yml)/[模型](https://github.com/JusperLee/IIANet/releases/download/v1.0.0/lrs2.zip) |
| IIANet | LRS3 | 18.3 | 18.5 | 3.28 | 3.1 | 18.6 | 110.11 ms | [配置](configs/LRS3-IIANet.yml)/[模型](https://github.com/JusperLee/IIANet/releases/download/v1.0.0/lrs3.zip) | 
| IIANet | VoxCeleb2 | 13.6 | 14.3 | 3.12 | 3.1 | 18.6 | 110.11 ms| [配置](configs/Vox2-IIANet.yml)/[模型](https://github.com/JusperLee/IIANet/releases/download/v1.0.0/vox2.zip) |

## 💥真实数据推理
如果你想测试网络视频，可以直接使用[`inference.py`](inference.py).
```shell
# Inference on a single video
# You can modify the video path in inference.py
python inference.py
```

## 📚训练

在开始训练之前，请修改[`configs`](configs)中的参数配置。

一个简单的训练配置示例：

```yaml
data_config:
  train_dir: DataPreProcess/LRS2/tr
  valid_dir: DataPreProcess/LRS2/cv
  test_dir: DataPreProcess/LRS2/tt
  n_src: 1
  sample_rate: 16000
  segment: 2.0
  normalize_audio: false
  batch_size: 3
  num_workers: 24
  pin_memory: true
  persistent_workers: false
```

使用以下命令开始训练：

```shell
python train.py --conf_dir configs/LRS2-IIANet.yml
python train.py --conf_dir configs/LRS3-IIANet.yml
python train.py --conf_dir configs/Vox2-IIANet.yml
```

## 📈测试/推理

要在一个或多个GPU上评估模型，请指定`CUDA_VISIBLE_DEVICES`、`dataset`、`model`和`checkpoint`：

```shell
python test.py --conf_dir checkpoints/lrs2/conf.yml
python test.py --conf_dir checkpoints/lrs3/conf.yml
python test.py --conf_dir checkpoints/vox2/conf.yml
```

对于单张图像的推理，请参考[`inference.py`](inference.py)。

## 💡未来工作

1. 验证IIANet在更大规模数据集（如AVSpeech）上的有效性和鲁棒性。
2. 进一步优化IIANet的架构和训练策略，以在降低计算成本的同时提高语音分离质量。
3. 探索IIANet在其他多模态任务中的应用，例如语音增强、说话人识别等。

## 📜引用

如果您觉得我们的工作对您有所帮助，请考虑引用：

```
@inproceedings{lee2024iianet,
  title={IIANet: An Intra- and Inter-Modality Attention Network for Audio-Visual Speech Separation}, 
  author={Kai Li and Runxuan Yang and Fuchun Sun and Xiaolin Hu},
  booktitle={International Conference on Machine Learning},
  year={2024}
}
```