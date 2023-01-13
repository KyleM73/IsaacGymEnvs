#!/bin/bash
task=Hsr
experimentName=HSR_first
checkpoint=runs/$experimentName/nn/$experimentName.pth
progressFile=runs/$experimentName/progress.txt

mkdir ~/Dropbox/UT/Experiments/$task/$experimentName

echo "Begin Training"
echo "Experiment:" $experimentName

## PHASE 1 -- 0 Humans
python train.py task=$task headless=True experiment=$experimentName

cp $checkpoint runs/$experimentName/nn/phase1.pth

viddir=$(ls -t runs/$task/videos/| head -1)

python train.py task=$task test=True headless=True experiment=$experimentName checkpoint=$checkpoint task.videoDir=$viddir num_envs=1 train.params.config.minibatch_size=32

echo phase 1: done > $progressFile

dropboxdir=~/Dropbox/UT/Experiments/$task/$experimentName/$viddir

cp $progressFile $dropboxdir 

## PHASE 2 -- 4 Humans
#python train.py task=Bumpybot headless=True task.env.asset.numHumans=4 train.params.config.max_epochs=10000 experiment=$experimentName checkpoint=$checkpoint task.env.episodeLength=1000 num_envs=256 train.params.config.minibatch_size=8192

#cp $checkpoint runs/$experimentName/nn/phase2.pth

#viddir=$(ls -t runs/Bumpybot/videos/| head -1)

#python train.py task=Bumpybot test=True headless=True task.env.asset.numHumans=4 experiment=$experimentName checkpoint=$checkpoint task.videoDir=$viddir num_envs=1 train.params.config.minibatch_size=32 task.env.episodeLength=1000

#echo phase 2: done >> $progressFile

#dropboxdir=~/Dropbox/UT/Experiments/$experimentName/$viddir

#cp $progressFile $dropboxdir 

## PHASE 3 -- 10 Humans
#python train.py task=Bumpybot headless=True task.env.asset.numHumans=10 train.params.config.max_epochs=4000 experiment=$experimentName checkpoint=$checkpoint task.env.episodeLength=1000 num_envs=256 train.params.config.minibatch_size=8192

#cp $checkpoint runs/$experimentName/nn/phase3.pth

#viddir=$(ls -t runs/Bumpybot/videos/| head -1)

#python train.py task=Bumpybot test=True headless=True task.env.asset.numHumans=10 experiment=$experimentName checkpoint=$checkpoint task.videoDir=$viddir num_envs=1 train.params.config.minibatch_size=32 task.env.episodeLength=1000

#echo phase 3: done >> $progressFile
#echo done > $progressFile

#dropboxdir=~/Dropbox/UT/Experiments/$experimentName/$viddir

#cp $progressFile $dropboxdir 

cp ./train.sh ~/Dropbox/UT/Experiments/$task/$experimentName

echo "Training Complete."


