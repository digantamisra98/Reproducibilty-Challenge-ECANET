<h1 align="center">Efficient Channel Attention:<br>Reproducibility Challenge 2020</h1>
<p align="center">CVPR 2020 <a href="https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_ECA-Net_Efficient_Channel_Attention_for_Deep_Convolutional_Neural_Networks_CVPR_2020_paper.html" target="_blank">(Official Paper)</a></p>

<p align="center">
    <img src="https://raw.githubusercontent.com/acervenky/animated-github-badges/master/assets/acbadge.gif" width = 20 />
    <a href="http://hits.dwyl.io/digantamisra98/Reproducibilty-Challenge-ECANET" alt="HitCount">
        <img src="http://hits.dwyl.io/digantamisra98/Reproducibilty-Challenge-ECANET.svg" /></a>
    <a href="https://arxiv.org/abs/1910.03151" alt="ArXiv">
        <img src="https://img.shields.io/badge/Paper-arXiv-blue.svg" /></a>
    <a href="https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_ECA-Net_Efficient_Channel_Attention_for_Deep_Convolutional_Neural_Networks_CVPR_2020_paper.html"                     alt="CVF">
          <img src="https://img.shields.io/badge/CVF-Page-purple.svg" /></a>
    <a href="https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_ECA-Net_Efficient_Channel_Attention_for_Deep_Convolutional_Neural_Networks_CVPR_2020_paper.pdf"                        alt="PDF">
          <img src="https://img.shields.io/badge/CVPR-PDF-neon.svg" /></a>
    <a href="https://openaccess.thecvf.com/content_CVPR_2020/supplemental/Wang_ECA-Net_Efficient_Channel_CVPR_2020_supplemental.pdf" alt="Supp">
          <img src="https://img.shields.io/badge/CVPR-Supp-pink.svg" /></a>
    <a href="https://www.youtube.com/watch?v=ipZ2AS1b0rI" alt="Video">
          <img src="https://img.shields.io/badge/CVPR-Video-maroon.svg" /></a>
    <a href="https://mybinder.org/v2/gh/digantamisra98/Reproducibilty-Challenge-ECANET/HEAD" alt="ArXiv">
        <img src="https://mybinder.org/badge_logo.svg" /></a>
    <a href="https://twitter.com/DigantaMisra1" alt="Twitter">
          <img src="https://img.shields.io/twitter/url/https/twitter.com/DigantaMisra1.svg?style=social&label=Follow%20%40DigantaMisra1" /></a>
    <a href="https://wandb.ai/diganta/ECANet-sweep?workspace=user-diganta" alt="Dashboard">
        <img src="https://img.shields.io/badge/WandB-Dashboard-gold.svg" /></a>
    <a href="https://wandb.ai/diganta/ECANet-sweep/reports/ECA-Net-Efficient-Channel-Attention-for-Deep-Convolutional-Neural-Networks-NeurIPS-Reproducibility-Challenge-2020--VmlldzozODU0NTM" alt="RC2020">
        <img src="https://img.shields.io/badge/WandB-Report1-yellow.svg" /></a>
    <a href="https://wandb.ai/diganta/ECANet-sweep/reports/Efficient-Channel-Attention--VmlldzozNzgwOTE" alt="Report">
        <img src="https://img.shields.io/badge/WandB-Report2-yellow.svg" /></a>
    <a href="https://colab.research.google.com/drive/1PHG4u_mkOnbge4RIzPjtfda1N-oiaDKI?usp=sharing" alt="Colab">
        <img src="https://colab.research.google.com/assets/colab-badge.svg" /></a>
    <a href="https://github.com/BangguWu/ECANet" alt="Report">
        <img src="https://img.shields.io/badge/Official-Repository-black.svg" /></a>
</p>

<p align="center">
    <img width="1000" src="figures/seg.png">
    </br>
    <em>Bounding Box and Segmentation Maps of ECANet-50-Mask-RCNN using samples from the test set of MS-COCO 2017 dataset.</em>
</p>

# Introduction

<p float="center">
    <img src="figures/eca_module.jpg" width="1000" alt="Struct.">
    <br>
    <em>Structural comparison of SE and ECA attention mechanism.</em>
</p>

Efficient Channel Attention (ECA) is a simple efficient extension of the popular Squeeze-and-Excitation Attention Mechanism, which is based on the foundation concept of Local Cross Channel Interaction (CCI). Instead of using fully-connected layers with reduction ratio bottleneck as in the case of SENets, ECANet uses an adaptive shared (across channels) 1D convolution kernel on the downsampled GAP *C* x 1 x 1 tensor. ECA is an equivalently plug and play module similar to SE attention mechanism and can be added anywhere in the blocks of a deep convolutional neural networks. Because of the shared 1D kernel, the parameter overhead and FLOPs cost added by ECA is significantly lower than that of SENets while achieving similar or superior performance owing to it's capabilities of constructing adaptive kernels. This work was accepted at the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020. 

## How to run:

### CIFAR-10:

<p float="center">
    <img src="figures/acc.png" width="415" alt="Accuracy.">
    <img src="figures/loss.png" width="415" alt="Loss.">
    <br>
    <em>Mean training curves of different attention mechanisms using ResNet-18 for CIFAR-10 training over 5 runs.</em>
</p>

### Sweeps:

<p align="left">
    <img width="1000" src="figures/sweeps_run.png">
    </br>
    <em>Hyper-parameter sweep run on Weights & Biases using a ResNet-18 on CIFAR-10.</em>
</p>

### ImageNet:


The Triplet Attention layer is implemented in `triplet_attention.py`. Since triplet attention is a dimentionality-preserving module, it can be inserted between convolutional layers in most stages of most networks. We recommend using the model definition provided here with our [imagenet training repo](https://github.com/LandskapeAI/imagenet) to use the fastest and most up-to-date training scripts.

However, this repository includes all the code required to recreate the experiments mentioned in the paper. This sections provides the instructions required to run these experiments. Imagenet training code is based on the official PyTorch example.

To train a model on ImageNet, run `train_imagenet.py` with the desired model architecture and the path to the ImageNet dataset:

### Simple Training

```bash
python train_imagenet.py -a resnet18 [imagenet-folder with train and val folders]
```

The default learning rate schedule starts at 0.1 and decays by a factor of 10 every 30 epochs. This is appropriate for ResNet and models with batch normalization, but too high for AlexNet and VGG. Use 0.01 as the initial learning rate for AlexNet or VGG:

```bash
python main.py -a alexnet --lr 0.01 [imagenet-folder with train and val folders]
```

Note, however, that we do not provide model defintions for AlexNet, VGG, etc. Only the ResNet family and MobileNetV2 are officially supported.

### Multi-processing Distributed Data Parallel Training

You should always use the NCCL backend for multi-processing distributed training since it currently provides the best distributed training performance.

#### Single node, multiple GPUs:

```bash
python train_imagenet.py -a resnet50 --dist-url 'tcp://127.0.0.1:FREEPORT' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 [imagenet-folder with train and val folders]
```

#### Multiple nodes:

Node 0:
```bash
python train_imagenet.py -a resnet50 --dist-url 'tcp://IP_OF_NODE0:FREEPORT' --dist-backend 'nccl' --multiprocessing-distributed --world-size 2 --rank 0 [imagenet-folder with train and val folders]
```

Node 1:
```bash
python train_imagenet.py -a resnet50 --dist-url 'tcp://IP_OF_NODE0:FREEPORT' --dist-backend 'nccl' --multiprocessing-distributed --world-size 2 --rank 1 [imagenet-folder with train and val folders]
```

### Usage

```
usage: train_imagenet.py  [-h] [--arch ARCH] [-j N] [--epochs N] [--start-epoch N] [-b N]
                          [--lr LR] [--momentum M] [--weight-decay W] [--print-freq N]
                          [--resume PATH] [-e] [--pretrained] [--world-size WORLD_SIZE]
                          [--rank RANK] [--dist-url DIST_URL]
                          [--dist-backend DIST_BACKEND] [--seed SEED] [--gpu GPU]
                          [--multiprocessing-distributed]
                          DIR

PyTorch ImageNet Training

positional arguments:
  DIR                   path to dataset

optional arguments:
  -h, --help            show this help message and exit
  --arch ARCH, -a ARCH  model architecture: alexnet | densenet121 |
                        densenet161 | densenet169 | densenet201 |
                        resnet101 | resnet152 | resnet18 | resnet34 |
                        resnet50 | squeezenet1_0 | squeezenet1_1 | vgg11 |
                        vgg11_bn | vgg13 | vgg13_bn | vgg16 | vgg16_bn | vgg19
                        | vgg19_bn (default: resnet18)
  -j N, --workers N     number of data loading workers (default: 4)
  --epochs N            number of total epochs to run
  --start-epoch N       manual epoch number (useful on restarts)
  -b N, --batch-size N  mini-batch size (default: 256), this is the total
                        batch size of all GPUs on the current node when using
                        Data Parallel or Distributed Data Parallel
  --lr LR, --learning-rate LR
                        initial learning rate
  --momentum M          momentum
  --weight-decay W, --wd W
                        weight decay (default: 1e-4)
  --print-freq N, -p N  print frequency (default: 10)
  --resume PATH         path to latest checkpoint (default: none)
  -e, --evaluate        evaluate model on validation set
  --pretrained          use pre-trained model
  --world-size WORLD_SIZE
                        number of nodes for distributed training
  --rank RANK           node rank for distributed training
  --dist-url DIST_URL   url used to set up distributed training
  --dist-backend DIST_BACKEND
                        distributed backend
  --seed SEED           seed for initializing training.
  --gpu GPU             GPU id to use.
  --multiprocessing-distributed
                        Use multi-processing distributed training to launch N
                        processes per node, which has N GPUs. This is the
                        fastest way to use PyTorch for either single node or
                        multi node data parallel training
```


### MS-COCO:

<p align="left">
    <img width="500" src="figures/seg_ep.gif">
    <br>
    <em>Training progress of ECANet-50-Mask-RCNN for 12 epochs.</em>
</p>

|Backbone|Detectors|BBox_AP|BBox_AP<sub>50</sub>|BBox_AP<sub>75</sub>|BBox_AP<sub>S</sub>|BBox_AP<sub>M</sub>|BBox_AP<sub>L</sub>|Segm_AP|Segm_AP<sub>50</sub>|Segm_AP<sub>75</sub>|Segm_AP<sub>S</sub>|Segm_AP<sub>M</sub>|Segm_AP<sub>L</sub>|Weights|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|ECANet-50|Mask RCNN|**34.1**|**53.4**|**37.0**|**21.1**|**37.2**|**42.9**|**31.4**|**50.6**|**33.2**|**18.1**|**34.3**|**41.1**|[Google Drive](https://drive.google.com/file/d/1IrxmSDDOzHKBJPkXYvCHNe-Koqm3Idtq/view?usp=sharing)|

## Cite:

```
@InProceedings{Wang_2020_CVPR,
author = {Wang, Qilong and Wu, Banggu and Zhu, Pengfei and Li, Peihua and Zuo, Wangmeng and Hu, Qinghua},
title = {ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks},
booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```

<p align="center">
    Made with ❤️ and ⚡
</p>
