#!/bin/sh

numHumans=3
epochs=10000
#epochs_humans=5000
saveVideo=True
checkpoint=runs/Bumpybot/nn/Bumpybot.pth
experimentName=Bumpybot_long

for arg in $@
do
	experimentName=$arg
done

echo "Begin Training"
echo "Experiment:" $experimentName

#python train.py task=Bumpybot headless=True task.env.asset.numHumans=$numHumans train.params.config.max_epochs=$epochs task.viewer.captureVideo=$saveVideo experiment=$experimentName task.image.fixCamera=True #checkpoint=$checkpoint

viddir=$(ls -t runs/Bumpybot/videos/| head -1)

#cp -r ./runs/Bumpybot/videos/$viddir ~/Dropbox/UT/Videos/

echo "Training Complete."


