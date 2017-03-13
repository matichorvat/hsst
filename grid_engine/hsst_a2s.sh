#!/bin/bash
#$ -S /bin/bash

hostname
echo "START: "`date`

p=$SGE_TASK_ID

config1=$1
config2=$2
tsttot=$3
numparts=$4

### gather tst ids to process
tpartsize=$[$tsttot / $numparts]
rest=$[$tsttot % $numparts]
if [[ $rest -gt 0 && $p -le $rest ]]; then
    lidx=$[($tpartsize + 1) * ($p - 1) + 1 ]
    uidx=$[$lidx + $tpartsize]
else
    lidx=$[$tpartsize * ($p - 1) + 1 + $rest]
    uidx=$[$lidx + $tpartsize - 1]
fi

echo "Files $lidx to $uidx"
set -x
/home/miproj/mphil.acs.oct2012/mh693/code/hsst/bin/alilats2splats.py -v $lidx-$uidx $config1 $config2

echo FINISHED: `date`
