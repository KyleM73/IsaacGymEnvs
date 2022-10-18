#!/bin/sh

numEnvs=16
numHumans=3
miniBatch=32
saveVideo=True
headless=False
experimentName=Bumpybot_long
checkpoint=runs/Bumpybot_long/nn/Bumpybot_long.pth #session2a.pth

for arg in $@
do
	checkpoint=$arg
done

python train.py test=True task=Bumpybot headless=$headless task.env.asset.numHumans=$numHumans task.viewer.captureVideo=$saveVideo experiment=$experimentName num_envs=$numEnvs train.params.config.minibatch_size=$miniBatch task.image.fixCamera=True checkpoint=$checkpoint

#if [$saveVideo] 
#then
#	echo "hi"
viddir=$(ls -t runs/Bumpybot/videos/| head -1)
echo $viddir
cp -r ./runs/Bumpybot/videos/$viddir ~/Dropbox/UT/Videos/
#fi
