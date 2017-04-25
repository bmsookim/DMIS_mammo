DDSM : Sub-project for DMIS digital mammography DREAM challenge
=======================================================================================================
Classification of DDSM dataset for DMIS-Dream Challenge
Feb 7, 2017 DMIS Digital Mammography DREAM Challenge.

Korea University, Data-Mining Lab
Bumsoo Kim (meliketoy@gmail.com)

# Mass-Classification

Torch Implementation for Daniel L'evy, Arzav Jain's
[Breast Mass Classification from Mammograms using Deep Convolutional Neural Networks](https://arxiv.org/pdf/1612.00542v1.pdf)

## Requirements
See the [installation instruction](INSTALL.md) for a step-by-step installation guide.
See the [server instruction](SERVER.md) for server setup.

- Install [Torch](http://torch.ch/docs/getting-started.html)
- Install [cuda-8.0](https://developer.nvidia.com/cuda-downloads)
- Install [cudnn v5.1](https://developer.nvidia.com/cudnn)
- Install luarocks packages
```bash
$ luarocks install cutorch
$ luarocks install xlua
$ luarocks install optnet
$ luarocks install nnlr
```

## Directories & Datasets
- modelState    : The best model will be saved in this directory
- datasets      : Data preparation & preprocessing directory
- augmented     : Data directory for augmented training & validation & test set
- networks      : Residual Network model structure file directory
- gen           : Generated t7 file for each dataset will be saved in this directory
- pretrained    : Pretrained networks will be downloaded in this directory

## Pretrained models
* [ResNet-18](https://d2j0dndfm35trm.cloudfront.net/resnet-18.t7)
* [ResNet-34](https://d2j0dndfm35trm.cloudfront.net/resnet-34.t7)
* [ResNet-50](https://d2j0dndfm35trm.cloudfront.net/resnet-50.t7)
* [ResNet-200](https://d2j0dndfm35trm.cloudfront.net/resnet-200.t7)
