#!/bin/bash
task=Bumpybot
experimentName=Bumpybot
checkpoint=runs/$experimentName/nn/$experimentName.pth
progressFile=runs/$experimentName/progress.txt

mkdir -p ~/Dropbox/UT/Experiments/$task/$experimentName

cp ./train.sh ~/Dropbox/UT/Experiments/$task/$experimentName/train.sh
cp ./cfg/task/$task.yaml ~/Dropbox/UT/Experiments/$task/$experimentName/$task.yaml
cp ./cfg/train/${task}PPO.yaml ~/Dropbox/UT/Experiments/$task/$experimentName/${task}PPO.yaml
f=`echo "$task" | sed 's/./\L&/g'`
cp ./tasks/$f.py ~/Dropbox/UT/Experiments/$task/$experimentName/$f.py

echo "Begin Training"
echo "Experiment:" $experimentName

## TRAIN
python train.py task=$task headless=True experiment=$experimentName

cp $checkpoint runs/$experimentName/nn/phase1.pth
cp $checkpoint ~/Dropbox/UT/Experiments/$task/$experimentName/model.pth

echo training: done > $progressFile

## TEST
viddir=$(ls -t runs/$task/videos/| head -1)

python train.py task=$task test=True headless=True experiment=$experimentName checkpoint=$checkpoint task.videoDir=$viddir num_envs=1 train.params.config.minibatch_size=20

echo testing: done > $progressFile

cp $progressFile ~/Dropbox/UT/Experiments/$task/$experimentName

echo "Training Complete."