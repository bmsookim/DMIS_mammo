Module #4 : ornot-mass
=======================================================================================================
Double-check of mass regions for DMIS-Dream Challenge
April 25, 2017 DMIS Digital Mammography DREAM Challenge.

Korea University, Data-Mining Lab
Bumsoo Kim (meliketoy@gmail.com)

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

## Download pretrained models
The following [prepare_pretrained.sh](./prepare_pretrained.sh) module will automatically download all the pre-trained models you need.
```bash
$ ./prepare_pretrained.sh
```

## Execute modes
- script_train.sh	: The training mode. The following will train over the pre-trained model of ILSVRC-2012.
- validate.sh		: The validation mode. The following will drop a 'result.csv' file in the format below.
```bash
[:filename],[:prediction score]
```

You can execute each modes running the code below
```bash
$ ./[:execute_mode].sh
```
