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

### MS-COCO:

<p align="left">
    <img width="500" src="figures/seg_ep.gif">
    <br>
    <em>Training progress of ECANet-50-Mask-RCNN for 12 epochs.</em>
</p>

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
