#!/bin/sh

experimentName=Bumpybot_dense
checkpoint=runs/$experimentName/nn/$experimentName.pth
progressFile=runs/Bumpybot/progress.txt

#for arg in $@
#do
#	experimentName=$arg
#done

echo "Begin Training"
echo "Experiment:" $experimentName

## PHASE 1 -- 0 Humans
python train.py task=Bumpybot headless=True task.env.asset.numHumans=0 train.params.config.max_epochs=250 task.viewer.captureVideo=True experiment=$experimentName task.image.fixCamera=True

viddir=$(ls -t runs/Bumpybot/videos/| head -1)

python train.py task=Bumpybot test=True headless=True task.env.asset.numHumans=0 task.viewer.captureVideo=True experiment=$experimentName task.image.fixCamera=True checkpoint=$checkpoint task.videoDir=$viddir num_envs=1 train.params.config.minibatch_size=32

echo done > $progressFile

dropboxdir=~/Dropbox/UT/Experiments/$viddir

cp $progressFile $dropboxdir 

## PHASE 2 -- 1 Human
python train.py task=Bumpybot headless=True task.env.asset.numHumans=1 train.params.config.max_epochs=500 task.viewer.captureVideo=True experiment=$experimentName task.image.fixCamera=True checkpoint=$checkpoint

viddir=$(ls -t runs/Bumpybot/videos/| head -1)

python train.py task=Bumpybot test=True headless=True task.env.asset.numHumans=1 task.viewer.captureVideo=True experiment=$experimentName task.image.fixCamera=True checkpoint=$checkpoint task.videoDir=$viddir num_envs=1 train.params.config.minibatch_size=32

echo done > $progressFile

dropboxdir=~/Dropbox/UT/Experiments/$viddir

cp $progressFile $dropboxdir 

## PHASE 3 -- 2 Humans
python train.py task=Bumpybot headless=True task.env.asset.numHumans=4 train.params.config.max_epochs=1500 task.viewer.captureVideo=True experiment=$experimentName task.image.fixCamera=True checkpoint=$checkpoint

viddir=$(ls -t runs/Bumpybot/videos/| head -1)

python train.py task=Bumpybot test=True headless=True task.env.asset.numHumans=4 task.viewer.captureVideo=True experiment=$experimentName task.image.fixCamera=True checkpoint=$checkpoint task.videoDir=$viddir num_envs=1 train.params.config.minibatch_size=32

echo done > $progressFile

dropboxdir=~/Dropbox/UT/Experiments/$viddir

cp $progressFile $dropboxdir 

## PHASE 4 -- 3 Humans
python train.py task=Bumpybot headless=True task.env.asset.numHumans=6 train.params.config.max_epochs=3000 task.viewer.captureVideo=True experiment=$experimentName task.image.fixCamera=True checkpoint=$checkpoint

viddir=$(ls -t runs/Bumpybot/videos/| head -1)

python train.py task=Bumpybot test=True headless=True task.env.asset.numHumans=6 task.viewer.captureVideo=True experiment=$experimentName task.image.fixCamera=True checkpoint=$checkpoint task.videoDir=$viddir num_envs=1 train.params.config.minibatch_size=32


echo done > $progressFile

dropboxdir=~/Dropbox/UT/Experiments/$viddir

cp $progressFile $dropboxdir 

echo "Training Complete."


