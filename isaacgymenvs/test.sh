#!/bin/sh

numEnvs=1
numHumans=0
miniBatch=32
saveVideo=True
headless=True
experimentName=Bumpybot_remote
checkpoint=runs/Bumpybot_remote/nn/Bumpybot_remote.pth #session2a.pth

for arg in $@
do
	checkpoint=$arg
done

python train.py test=True task=Bumpybot headless=$headless task.env.asset.numHumans=$numHumans task.viewer.captureVideo=$saveVideo experiment=$experimentName num_envs=$numEnvs train.params.config.minibatch_size=$miniBatch task.image.fixCamera=True checkpoint=$checkpoint

#if [$saveVideo] 
#then
#	echo "hi"
viddir=$(ls -t runs/Bumpybot/videos/| head -1)
#echo $viddir
cp -r ./runs/Bumpybot/videos/$viddir ~/Dropbox/UT/Experiments/
#fi
