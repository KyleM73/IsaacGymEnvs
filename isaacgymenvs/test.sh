#!/bin/sh

numEnvs=1
numHumans=0
miniBatch=32
saveVideo=True
headless=False
experimentName=Bumpybot
checkpoint=runs/Bumpybot/nn/Bumpybot.pth

for arg in $@
do
	checkpoint=$arg
done

python train.py test=True task=Bumpybot headless=$headless task.env.asset.numHumans=$numHumans task.viewer.captureVideo=$saveVideo experiment=$experimentName num_envs=$numEnvs train.params.config.minibatch_size=$miniBatch checkpoint=$checkpoint task.image.fixCamera=True 