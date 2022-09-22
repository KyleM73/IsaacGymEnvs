#!/bin/sh

numHumans=2
epochs=8000
epochs_humans=5000
saveVideo=True
checkpoint=runs/Bumpybot/nn/Bumpybot.pth
experimentName=Bumpybot

for arg in $@
do
	experimentName=$arg
done

echo "Begin Training"
echo "Experiment:" $experimentName

## PHASE 1
python train.py task=Bumpybot headless=True task.env.asset.numHumans=0 train.params.config.max_epochs=$epochs task.viewer.captureVideo=$saveVideo experiment=$experimentName task.image.fixCamera=True 

## CLEAR MEMORY
#python clear_mem.py

## PHASE 2
#python train.py task=Bumpybot headless=True task.env.asset.numHumans=0 train.params.config.max_epochs=$epochs task.viewer.captureVideo=$saveVideo experiment=$experimentName task.image.fixCamera=False

## CLEAR MEMORY
#python clear_mem.py

## PHASE 3
#python train.py task=Bumpybot headless=True checkpoint=$checkpoint task.env.asset.numHumans=$numHumans train.params.config.max_epochs=$epochs_humans task.viewer.captureVideo=$saveVideo experiment=$experimentName

echo "Training Complete."

## TEST

#./test.sh


