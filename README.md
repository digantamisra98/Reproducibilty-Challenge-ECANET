<h1 align="center">Efficient Channel Attention:<br>Reproducibility Challenge 2020</h1>
<p align="center">CVPR 2020 <a href="https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_ECA-Net_Efficient_Channel_Attention_for_Deep_Convolutional_Neural_Networks_CVPR_2020_paper.html" target="_blank">(Official Paper)</a></p>

<p align="center">
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
    <br>
    <a href="https://wandb.ai/diganta/ECANet-sweep?workspace=user-diganta" alt="Dashboard">
        <img src="https://img.shields.io/badge/WandB-Dashboard-gold.svg" /></a>
    <a href="https://wandb.ai/diganta/ECANet-sweep/reports/ECA-Net-Efficient-Channel-Attention-for-Deep-Convolutional-Neural-Networks-NeurIPS-Reproducibility-Challenge-2020--VmlldzozODU0NTM" alt="RC2020">
        <img src="https://img.shields.io/badge/WandB-Report1-yellow.svg" /></a>
    <a href="https://wandb.ai/diganta/ECANet-sweep/reports/Efficient-Channel-Attention--VmlldzozNzgwOTE" alt="Report">
        <img src="https://img.shields.io/badge/WandB-Report2-yellow.svg" /></a>
    <a href="https://github.com/BangguWu/ECANet" alt="Report">
        <img src="https://img.shields.io/badge/Official-Repository-black.svg" /></a>
    <a href="https://blog.paperspace.com/attention-mechanisms-in-computer-vision-ecanet/" alt="Report">
        <img src="https://img.shields.io/badge/Paperspace-Blog-white.svg" /></a>
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

#### Install Dependencies:

```
pip install -r requirements.txt
```

This reproduction is build on PyTorch and MMDetection. Ensure you have CUDA Toolkit > 10.1 installed. For more details regarding installation of MMDetection, please visit this [resources page](https://mmdetection.readthedocs.io/en/latest/get_started.html#installation).

If ```pip install mmcv-full``` takes a lot of time or fails, use the following line (customize the torch and cuda versions as per your requirements):
```
pip install mmcv-full==latest+torch1.7.0+cu101 -f https://download.openmmlab.com/mmcv/dist/index.html
```

Although [Echo](https://github.com/digantamisra98/Echo) can be installed pip, the features we currently use in this project aren't available in the latest pip version. So it's advisable to rather install from source by the following commands and then clone this repository within the directory where Echo source is present and installed in your environment/local/instance:

```
import os
git clone https://github.com/digantamisra98/Echo.git
os.chdir("/path_to_Echo")
git clone https://github.com/digantamisra98/ECANet.git
pip install -e "/path_to_Echo/"
```

### CIFAR-10:

<p>
        <a href="https://colab.research.google.com/drive/1ssmRtF8U4-6NtZoXLZ7uNSu0g1skHPTT?usp=sharing" alt="Colab">
        <img src="https://colab.research.google.com/assets/colab-badge.svg" /></a>
</p>

<p float="center">
    <img src="figures/acc.png" width="415" alt="Accuracy.">
    <img src="figures/loss.png" width="415" alt="Loss.">
    <br>
    <em>Mean training curves of different attention mechanisms using ResNet-18 for CIFAR-10 training over 5 runs.</em>
</p>

Using the above linked colab notebook, you can run comparative runs for different attention mechanisms on CIFAR-10 using ResNets. You can add your own attention mechanisms by adding them in the source of Echo package.

### Sweeps:

<p>
        <a href="https://colab.research.google.com/drive/1LfZWJOrxpovPAbQOKDi-6cgXTLdmRYTG?usp=sharing" alt="Colab">
        <img src="https://colab.research.google.com/assets/colab-badge.svg" /></a>
        <a href="https://wandb.ai/diganta/ECANet-sweep/sweeps/z61h01i4?workspace=user-diganta" alt="Dashboard">
        <img src="https://img.shields.io/badge/Sweeps-Dashboard-gold.svg" /></a>
</p>

<p align="left">
    <img width="1000" src="figures/sweeps_run.png">
    </br>
    <em>Hyper-parameter sweep run on Weights & Biases using a ResNet-18 on CIFAR-10.</em>
</p>

To run hyperparamter sweeps on [WandB](https://wandb.ai/site), simply run the above linked colab notebook. To add more hyperparameters, simply edit the `sweep.yaml` file present in `sweep` folder.

### ImageNet:

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/eca-net-efficient-channel-attention-for-deep/image-classification-on-imagenet)](https://paperswithcode.com/sota/image-classification-on-imagenet?p=eca-net-efficient-channel-attention-for-deep)

ECA layer is implemented in [eca_module.py](https://github.com/digantamisra98/Reproducibilty-Challenge-ECANET/blob/main/models/eca_module.py). Since ECA is a dimentionality-preserving module, it can be inserted between convolutional layers in most stages of most networks. We recommend using the model definition provided here with our [imagenet training repo](https://github.com/LandskapeAI/imagenet) to use the fastest and most up-to-date training scripts along with detailed instructions on how to download and prepare dataset.

#### Train with ResNet

You can run the `main.py` to train or evaluate as follow:

```
CUDA_VISIBLE_DEVICES={device_ids} python main -a {model_name} --project {WandB Project Name} {the path of you datasets}
```
For example:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main -a eca_resnet50 --project ECANet_RC2020 ./datasets/ILSVRC2012/images
```

#### Train with MobileNet_v2
It is same with above ResNet replace `main.py` by `light_main.py`.

#### Compute the parameters and FLOPs
If you have install [thop](https://github.com/Lyken17/pytorch-OpCounter), you can `paras_flosp.py` to compute the parameters and FLOPs of our models. The usage is below:
```
python paras_flops.py -a {model_name}
```

##### Official Results: 

|Model|Param.|FLOPs|Top-1(%)|Top-5(%)|
|:---:|:----:|:---:|:------:|:------:|
|ECA-Net18|11.15M|1.70G|70.92|89.93|
|ECA-Net34|20.79M|3.43G|74.21|91.83|
|ECA-Net50|24.37M|3.86G|77.42|93.62|
|ECA-Net101|42.49M|7.35G|78.65|94.34|
|ECA-Net152|57.41M|10.83G|78.92|94.55|
|ECA-MobileNet_v2|3.34M|319.9M|72.56|90.81||

### MS-COCO:

<p align="left">
    <img width="500" src="figures/seg_ep.gif">
    <br>
    <em>Training progress of ECANet-50-Mask-RCNN for 12 epochs.</em>
</p>

##### Reproduced Results:

|Backbone|Detectors|BBox_AP|BBox_AP<sub>50</sub>|BBox_AP<sub>75</sub>|BBox_AP<sub>S</sub>|BBox_AP<sub>M</sub>|BBox_AP<sub>L</sub>|Segm_AP|Segm_AP<sub>50</sub>|Segm_AP<sub>75</sub>|Segm_AP<sub>S</sub>|Segm_AP<sub>M</sub>|Segm_AP<sub>L</sub>|Weights|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|ECANet-50|Mask RCNN|**34.1**|**53.4**|**37.0**|**21.1**|**37.2**|**42.9**|**31.4**|**50.6**|**33.2**|**18.1**|**34.3**|**41.1**|[Google Drive](https://drive.google.com/file/d/1IrxmSDDOzHKBJPkXYvCHNe-Koqm3Idtq/view?usp=sharing)|

#### Download MS-COCO 2017:

Simply execute [this script](https://gist.githubusercontent.com/mkocabas/a6177fc00315403d31572e17700d7fd9/raw/a6ad5e9d7567187b65f222115dffcb4b8667e047/coco.sh) in your terminal to download and process the MS-COCO 2017 dataset. You can use the following command to do the same:
```
curl https://gist.githubusercontent.com/mkocabas/a6177fc00315403d31572e17700d7fd9/raw/a6ad5e9d7567187b65f222115dffcb4b8667e047/coco.sh | sh
```
#### Download Pretrained ImageNet Weights:

Download the pretrained weights from the [original repository](https://github.com/BangguWu/ECANet). You can download them using `gdown` if you're on Colab or GCloud. For example to download the ECANet-50 weights for training a Mask RCNN, use the following command:

```
pip install gdown
gdown https://drive.google.com/u/0/uc?id=1670rce333c_lyMWFzBlNZoVUvtxbCF_U&export=download
```

To make the weights compatible for MS-COCO training, run [this notebook](https://github.com/digantamisra98/Reproducibilty-Challenge-ECANET/blob/main/Weight_correction.ipynb) and then move the processed weight file `eca_net.pth.tar` to a new folder named `weights` in mmdetection directory. Once done, edit the `model` dict variable in `mmdetection/configs/_base_/models/mask_rcnn_r50_fpn.py` by updating the `pretrained` parameter to ```pretrained='weights/eca_net.pth.tar'```. This will load the ECANet-50 backbone weights correctly. 

#### Training:

This project uses [MMDetection](https://github.com/open-mmlab/mmdetection) for training the Mask RCNN model. One would require to make the following changes in the following file in the cloned source of MMDetection codebase to train the detector model.

- `mmdetection/mmdet/models/backbones/resnet.py`:
    All that requires to be done now is to modify the source backbone code to convert it into ECA based backbone. For this case, the backbone is ECANet-50 and the detector is Mask-RCNN. Simply go to this file and add the original class definition of ECA Module which is:
    
    ```
    class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)
    ```
    
    Once done, in the `__init__` function of class `Bottleneck`, add the following code lines:
    
    ```
    if self.planes == 64:
            self.eca = eca_layer(k_size = 3)
    elif self.planes == 128:
        self.eca = eca_layer(k_size = 5)
    elif self.planes == 256:
        self.eca = eca_layer(k_size = 5)
    elif self.planes == 512:
        self.eca = eca_layer(k_size = 7)
    ```
    
    *Note: This is done to ensure the backbone weights get loaded properly as ECANet-50 uses the input number of channels of the block <b>C</b> to predefine the kernel size for the 1D convolution filter in the ECA Module.*
    
    Lastly, just add the following line to the `forward` pass/ function of the same class right after the final conv + normalization layer:
    ```
    out = self.eca(out)
    ```
    
- `mmdetection/configs/_base_/schedules/schedule_1x.py`
    If you're training on 1 GPU, you would require to lower down the LR for the scheduler since MMDetection default LR strategy is set for 8 GPU based training. Simply go to this file and edit the optimizer definition with the lr value now being `0.0025`.

After making the following changes to run the training, use the following command:
```
python tools/train.py mmdetection/configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py
```

To resume training from any checkpoint, use the following command (for example - Epoch 5 in this case):
```
python tools/train.py configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py --resume-from work_dirs/mask_rcnn_r50_fpn_1x_coco/epoch_5.pth
```

#### Inference:

To run inference, simply run [this notebook](https://github.com/digantamisra98/Reproducibilty-Challenge-ECANET/blob/main/inference_demo.ipynb).
*Although the authors provide the trained detector weights in their repository, they contain a lot of bugs which are described in this [open issue](https://github.com/BangguWu/ECANet/issues).*

##### Logs:

The logs are provided in the [Logs folder](https://github.com/digantamisra98/Reproducibilty-Challenge-ECANET/tree/main/logs). It contains two files:
1. ```20210102_160817.log```: Contains logs from epoch 1 to epoch 6
2. ```20210106_012255.log```: Contains logs from epoch 6 to epoch 12
I restarted training from epoch 6 again since the lr was on 8 GPU setting while I was training on 1 GPU which caused nan loss at epoch 6, hence the two log files.

## WandB logs:

The dashboard for this project can be accessed [here](https://wandb.ai/diganta/ECANet-sweep?workspace=user-diganta).

##### Machine Specifications and Software versions:

- torch: 1.7.1+cu110
- GPU: 1 NVIDA V100, 16GB Memory on GCP

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
