#/bin/bash
export netType='wide-resnet'
export depth=10
export width=5
export dataset='dreamChallenge'
export data='../../dataset/CALCIFICATION/'

rm -rf gen/dreamChallenge.t7

th main.lua \
    -dataset ${dataset} \
    -data ${data} \
    -netType ${netType} \
    -nGPU 2 \
    -batchSize 64 \
    -LR 1e-1 \
    -weightDecay 1e-3 \
    -depth ${depth} \
    -widen_factor ${width} \
    -resetClassifier true \
    -nClasses 2 \
    -optnet true \
