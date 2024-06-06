[简体中文](README_zh-CN.md) | English

# <font color=E7595C>I</font><font color=F6C446>I</font><font color=00C7EE>A</font><font color=00D465>Net</font>: An <font color=E7595C>I</font>ntra- and <font color=F6C446>I</font>nter-Modality <font color=00C7EE>A</font>ttention <font color=00D465>Net</font>work for Audio-Visual Speech Separation

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/scanet-a-self-and-cross-attention-network-for/speech-separation-on-lrs2)](https://paperswithcode.com/sota/speech-separation-on-lrs2?p=scanet-a-self-and-cross-attention-network-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/scanet-a-self-and-cross-attention-network-for/speech-separation-on-lrs3)](https://paperswithcode.com/sota/speech-separation-on-lrs3?p=scanet-a-self-and-cross-attention-network-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/scanet-a-self-and-cross-attention-network-for/speech-separation-on-voxceleb2)](https://paperswithcode.com/sota/speech-separation-on-voxceleb2?p=scanet-a-self-and-cross-attention-network-for)
[![arXiv](https://img.shields.io/badge/arXiv-2308.08143-b31b1b.svg)](https://arxiv.org/abs/2308.08143)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://makeapullrequest.com) 
[![GitHub license](https://img.shields.io/github/license/JusperLee/IIANet.svg?color=blue)](https://github.com/JusperLee/IIANet/blob/master/LICENSE)
![GitHub stars](https://img.shields.io/github/stars/JusperLee/IIANet)
![GitHub forks](https://img.shields.io/github/forks/JusperLee/IIANet)
![Website](https://img.shields.io/website?url=https%3A%2F%2Fcslikai.cn%2FIIANet%2F&up_message=Demo%20Page&down_message=Demo%20Page&logo=webmin)


By [1] Tsinghua University, [2]Chinese Institute for Brain Research.
* [Kai Li](https://cslikai.cn)[1], Runxuan Yang[1], [Fuchun Sun](https://scholar.google.com/citations?user=DbviELoAAAAJ&hl=en)[1], [Xiaolin Hu](https://www.xlhu.cn/)[1,2].

This repository is an official implementation of the IIANet accepted to **ICML 2024** (**Poster**).

## ✨Key Highlights:

1. We propose an attention-based cross-modal speech separation network called IIANet, which extensively uses intra-attention (IntraA) and inter-attention (InterA) mechanisms within and across the speech and video modalities.

2. Compared with existing CNN and Transformer methods, IIANet achieves significantly better separation quality on three audio-visual speech separation datasets while greatly reducing computational complexity and memory usage.

3. A faster version, IIANet-fast, surpasses CTCNet by 1.1 dB on the challenging LRS2 dataset with only 11% MACs of CTCNet.

4. Qualitative evaluations on real-world YouTube scenarios show that IIANet generates higher-quality separated speech than other separation models.

## 🚀Overall Pipeline

<video playsinline="" autoplay="" loop="" preload="" muted="" width="900">
                <source src="figures/overall.mp4" type="video/mp4">
              </video>

## 🪢IIANet Architecture

<video playsinline="" autoplay="" loop="" preload="" muted="" width="900">
                <source src="figures/separation.mp4" type="video/mp4">
              </video>

## 🔧Installation

1. Clone the repository:

```shell
git clone https://github.com/JusperLee/IIANet.git 
cd IIANet/
```

2. Create and activate the conda environment:

```shell
conda create -n iianet python=3.8 
conda activate iianet
```

3. Install PyTorch and torchvision following the [official instructions](https://pytorch.org). The code requires `python>=3.8`, `pytorch>=1.11`, `torchvision>=0.13`.

4. Install other dependencies:

```shell 
pip install -r requirements.txt
```

## 📊Model Performance  

We evaluate IIANet and its fast version IIANet-fast on three datasets: LRS2, LRS3, and VoxCeleb2. The results show that IIANet achieves significantly better speech separation quality than existing methods while maintaining high efficiency [1].

| Method | Dataset | SI-SNRi | SDRi | PESQ | Params | MACs | GPU Infer Time | Download |
|:---:|:-----:|:------:|:----:|:----:|:------:|:-----:|:-----------:|:----:|  
| IIANet | LRS2 | 16.0 | 16.2 | 3.23 | 3.1 | 18.6 | 110.11 ms | [Config](configs/LRS2-IIANet.yml)/[Model](https://github.com/JusperLee/IIANet/releases/download/v1.0.0/lrs2.zip) |
| IIANet | LRS3 | 18.3 | 18.5 | 3.28 | 3.1 | 18.6 | 110.11 ms | [Config](configs/LRS3-IIANet.yml)/[Model](https://github.com/JusperLee/IIANet/releases/download/v1.0.0/lrs3.zip) | 
| IIANet | VoxCeleb2 | 13.6 | 14.3 | 3.12 | 3.1 | 18.6 | 110.11 ms| [Config](configs/Vox2-IIANet.yml)/[Model](https://github.com/JusperLee/IIANet/releases/download/v1.0.0/vox2.zip) |

## 💥Real-world Evaluation
For single video inference, please refer to [`inference.py`](inference.py).
```shell
# Inference on a single video
# You can modify the video path in inference.py
python inference.py
```

## 📚Training

Before starting training, please modify the parameter configurations in [`configs`](configs).

A simple example of training configuration:

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

Use the following commands to start training:

```shell
python train.py --conf_dir configs/LRS2-IIANet.yml
python train.py --conf_dir configs/LRS3-IIANet.yml
python train.py --conf_dir configs/Vox2-IIANet.yml
```

## 📈Testing/Inference

To evaluate a model on one or more GPUs, specify the `CUDA_VISIBLE_DEVICES`, `dataset`, `model` and `checkpoint`:

```shell
python test.py --conf_dir checkpoints/lrs2/conf.yml
python test.py --conf_dir checkpoints/lrs3/conf.yml
python test.py --conf_dir checkpoints/vox2/conf.yml
```

## 💡Future Work

1. Validate the effectiveness and robustness of IIANet on larger-scale datasets such as AVSpeech.  
2. Further optimize the architecture and training strategies of IIANet to improve speech separation quality while reducing computational costs.
3. Explore the applications of IIANet in other multimodal tasks, such as speech enhancement, speaker recognition, etc.

## 📜Citation

If you find our work helpful, please consider citing:

```
@inproceedings{lee2024iianet,
  title={IIANet: An Intra- and Inter-Modality Attention Network for Audio-Visual Speech Separation}, 
  author={Kai Li and Runxuan Yang and Fuchun Sun and Xiaolin Hu},
  booktitle={International Conference on Machine Learning},
  year={2024}
}
```