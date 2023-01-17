#!/bin/bash
task=Bumpybot
experimentName=$(ls -t ~/Dropbox/UT/Experiments/$task/| head -1)
dropboxdir=~/Dropbox/UT/Experiments/$task/$experimentName
localdir=runs/$task/videos
videodir=$(ls -t $localdir/| head -1)

numPhases=1

for i in $(eval echo "{1..$numPhases}")
do
    echo recording from directory: $videodir
    flag=1

    while [ $flag -eq 1 ]
    do  
        cp -r $localdir/$videodir $dropboxdir

        if $([ -f $dropboxdir/progress.txt ])
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
