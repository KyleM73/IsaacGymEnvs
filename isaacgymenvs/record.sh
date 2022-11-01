#!/bin/bash

localdir=runs/Bumpybot/videos
videodir=$(ls -t $localdir/| head -1)
experimentName=$(ls -t ~/Dropbox/UT/Experiments/| head -1)
dropboxdir=~/Dropbox/UT/Experiments/$experimentName
numPhases=2

for i in $(eval echo "{1..$numPhases}")
do
    echo recording from directory: $videodir
    flag=1

    while [ $flag -eq 1 ]
    do  
        cp -r $localdir/$videodir $dropboxdir

        if $([ -f $dropboxdir/$videodir/progress.txt ])
        then
            flag=0
        fi
    
        sleep 60
    done

    echo recording from directory $videodir complete.

    sleep 300

    videodir=$(ls -t $localdir/| head -1)

done

echo recording complete.
