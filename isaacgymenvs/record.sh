#!/bin/sh

localdir=runs/Bumpybot/videos
videodir=$(ls -t $localdir/| head -1)
dropboxdir=~/Dropbox/UT/Experiments

echo recording from directory: $videodir

while true
do  
    cp -r $localdir/$videodir $dropboxdir

    if $([ -f $dropboxdir/$videodir/progress.txt ])
    then
        break
    fi
    
    sleep 60
done

sleep 300

localdir=runs/Bumpybot/videos
videodir=$(ls -t $localdir/| head -1)
dropboxdir=~/Dropbox/UT/Experiments

echo recording from directory: $videodir

while true
do  
    cp -r $localdir/$videodir $dropboxdir

    if $([ -f $dropboxdir/$videodir/progress.txt ])
    then
        break
    fi
    
    sleep 60
done

sleep 300

localdir=runs/Bumpybot/videos
videodir=$(ls -t $localdir/| head -1)
dropboxdir=~/Dropbox/UT/Experiments

echo recording from directory: $videodir

while true
do  
    cp -r $localdir/$videodir $dropboxdir

    if $([ -f $dropboxdir/$videodir/progress.txt ])
    then
        break
    fi
    
    sleep 60
done

sleep 300

localdir=runs/Bumpybot/videos
videodir=$(ls -t $localdir/| head -1)
dropboxdir=~/Dropbox/UT/Experiments

echo recording from directory: $videodir

while true
do  
    cp -r $localdir/$videodir $dropboxdir

    if $([ -f $dropboxdir/$videodir/progress.txt ])
    then
        break
    fi
    
    sleep 60
done

echo recording complete.
