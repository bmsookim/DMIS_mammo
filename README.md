DMIS-mammo
==================================================================================================
Repository for the 2017 Digital Mammography DREAM Challenge

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
```

## Directions and datasets
- Calcification-Classifier	: Classification whether the ROI region of calcification is malignant/benign.
- Mass-Classifier		: Classification whether the ROI region of mass is malignant/benign.
- ornot-Calcification		: Deciding whether the given window is a calcification ROI.
- ornot-Mass			: Deciding whether the given window is a mass ROI.

## Description of each modules

### 1. calcificatoin-classifier

The input of the [calcification-classifier](./calcification-classifier/) will be a square window of the ROI region of 'calcification'.
ROI regions will be extracted according to the heatmap derived from [ornot_calcification](./ornot_calcification).

- Input size : 256 x 256
- Crop size  : 224 x 224
- Model      : Fine-tuned Residual Network 50 (ILSVRC-2012)
- Best acc   : 80%

### 2. mass-classifier

The input of the [mass-classifier](./mass-classifier/) will be a square window of the ROI region of 'mass'.
ROI regions will be extracted according to
```bash
Total score = (Faster-RCNN results) + (Distance comparison of CC, MLO views) + ([ornot-Mass](./ornot-Mass/) results)
```
of the 'mass' regions in our private dataset.

- Input size : 256 x 256
- Crop size  : 224 x 224
- Model      : Fine-tuned Residual Network 50 (ILSVRC-2012)
- Best acc   : None

### 3. ornot-calcification

The input of the [ornot-calcification](./ornot-calcification/) will be a tiny window of the suspected region of 'calcification'.
Regions will be extracted according to Faster-RCNN training of the 'calcification' regions in our private dataset.

- Input size : 36 x 36
- Crop size  : 32 x 32
- Model      : Wide-Residual-Network 28x10
- Best acc   : 95.5%

### 4. ornot-mass

The input of the [ornot-mass](./ornot-mass/) will be a square window of the suspected region of 'mass'.
Regions will be extracted according to Faster-RCNN training of the 'Mass' regions in our private dataset.

- Input size : 256 x 256
- Crop size  : 224 x 224
- Model      : Fine-tuned Residual Network 50 (ILSVRC-2012)
- Best acc   : 83%
