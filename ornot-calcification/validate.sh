#!/bin/bash
export netType='wide-resnet'
export depth=28
export width=2
export dataset='dreamChallenge'
export data='CALCIFICATION/'

# rm -rf gen/dreamChallenge.t7

th test.lua \
    -dataset ${dataset} \
    -data ${data} \
    -netType ${netType} \
    -nGPU 2 \
    -batchSize 64 \
    -LR 1e-2 \
    -weightDecay 5e-4 \
    -depth ${depth} \
    -widen_factor ${width} \
    -resume modelState \
    -optnet true \
