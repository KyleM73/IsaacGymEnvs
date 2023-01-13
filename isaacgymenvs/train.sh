#!/bin/bash
task=Bumpybot
experimentName=Bumpybot
checkpoint=runs/$experimentName/nn/$experimentName.pth
progressFile=runs/$experimentName/progress.txt

mkdir ~/Dropbox/UT/Experiments/$task/$experimentName

echo "Begin Training"
echo "Experiment:" $experimentName

## PHASE 1 -- 3 Humans
python train.py task=$task headless=True experiment=$experimentName

cp $checkpoint runs/$experimentName/nn/phase1.pth

viddir=$(ls -t runs/$task/videos/| head -1)

python train.py task=$task test=True headless=True experiment=$experimentName checkpoint=$checkpoint task.videoDir=$viddir num_envs=1 train.params.config.minibatch_size=20

echo phase 1: done > $progressFile

dropboxdir=~/Dropbox/UT/Experiments/$task/$experimentName/$viddir

cp $progressFile $dropboxdir 

cp ./train.sh ~/Dropbox/UT/Experiments/$task/$experimentName

echo "Training Complete."


