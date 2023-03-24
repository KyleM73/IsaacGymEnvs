#!/bin/bash

if [ $# -eq 0 ]
then
    numTests=10
    numHumans=5
elif [ $# -eq 1 ]
then
    numTests=10
    numHumans=$1
else
    numTests=$2
    numHumans=$1
fi
echo Running $numTests tests with $numHumans humans

task=Bumpybot_blind
experimentName=Bumpybot_blind
#checkpoint=runs/$experimentName/nn/$experimentName.pth
checkpoint=runs/$experimentName/nn/last_Bumpybot_blind_ep_10000_rew_905.60657.pth

dt=$(date '+%m_%d_%Y-%H_%M')
viddir=test_$dt
for i in $(seq $numTests)
do
    saveviddir=$viddir/test$(($i-1))
    echo Saving to dir $saveviddir
    python train.py test=True task=$task headless=True task.env.asset.numHumans=$numHumans experiment=$experimentName num_envs=1 train.params.config.minibatch_size=20 checkpoint=$checkpoint task.videoDir=$viddir task.viewer.fancyTest.test=True
    mv runs/$task/videos/$viddir/test runs/$task/videos/$viddir/test$(($i-1))
done

cp -r runs/$task/videos/$viddir ~/Dropbox/UT/Experiments/$task/$experimentName
