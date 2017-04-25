#!/bin/bash
export netType='resnet'
export depth=50
export dataset='dreamChallenge'
export data='/scratch/KUMC-anamFUll-guroBenign/patch_model'
#export data='/scratch/KUMC-anamFUll-guroBenign/patch_model'

# rm -rf gen/dreamChallenge.t7

th extract_features.lua \
    -dataset ${dataset} \
    -data ${data} \
    -netType ${netType} \
    -nGPU 2 \
    -batchSize 64 \
    -LR 1e-2 \
    -weightDecay 1e-4 \
    -depth ${depth} \
    -resume modelState \
