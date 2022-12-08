#!/bin/bash

experimentName=Bumpybot_small_rew
checkpoint=runs/$experimentName/nn/$experimentName.pth
progressFile=runs/Bumpybot/progress.txt

mkdir ~/Dropbox/UT/Experiments/$experimentName

echo "Begin Training"
echo "Experiment:" $experimentName

## PHASE 1 -- 0 Humans
python train.py task=Bumpybot headless=True experiment=$experimentName task.env.asset.numHumans=5 train.params.config.max_epochs=2000 num_envs=256 train.params.config.minibatch_size=8192 

cp $checkpoint runs/$experimentName/nn/simple3.pth

viddir=$(ls -t runs/Bumpybot/videos/| head -1)

python train.py task=Bumpybot test=True headless=True experiment=$experimentName checkpoint=$checkpoint task.videoDir=$viddir num_envs=1 train.params.config.minibatch_size=32 task.env.asset.numHumans=5

echo training: done > $progressFile

dropboxdir=~/Dropbox/UT/Experiments/$experimentName/$viddir

cp $progressFile $dropboxdir 

cp ./simple_train.sh ~/Dropbox/UT/Experiments/$experimentName/simple_train3.sh

echo "Training Complete."


