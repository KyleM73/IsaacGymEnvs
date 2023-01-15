#!/bin/sh

if [ $# -eq 0 ]
then
    numTests=1
    numHumans=5
elif [ $# -eq 1 ]
then
    numTests=1
    numHumans=$1
else
    numTests=$2
    numHumans=$1
fi
echo Running $numTests tests with $numHumans humans

task=Bumpybot
experimentName=Bumpybot
checkpoint=runs/$experimentName/nn/$experimentName.pth
batchSize=$(($numTests*20))

dt='date '+%m_%d_%Y-%H_%M''
viddir=runs/$task/videos/test_$dt
for i in {1..$numTests}
do
    saveviddir=$viddir/test$i
    python train.py test=True task=Bumpybot headless=True task.env.asset.numHumans=$numHumans experiment=$experimentName num_envs=$numEnvs train.params.config.minibatch_size=$batchSize checkpoint=$checkpoint task.videoDir=$saveviddir
done

cp -r $saveviddir ~/Dropbox/UT/Experiments/%task/$experimentName
