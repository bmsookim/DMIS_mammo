#!/bin/bash
export netType='resnet'
export depth=50
export dataset='dreamChallenge'
export data='../../dataset/scratch/cal/'
#export data='/scratch/KUMC-anamFUll-guroBenign/patch_model'

# rm -rf gen/dreamChallenge.t7

th test.lua \
    -dataset ${dataset} \
    -data ${data} \
    -netType ${netType} \
    -nGPU 2 \
    -batchSize 64 \
    -LR 1e-2 \
    -weightDecay 1e-4 \
    -depth ${depth} \
    -resume Synapse \
    -resetClassifier true \
    -nClasses 2 \
    -nGPU 1 \
    -optnet false \
 
