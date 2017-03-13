#!/bin/bash
#$ -S /bin/bash

hostname
echo "START: "`date`

p=$SGE_TASK_ID

config=$1
tsttot=$2
numparts=$3

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
/home/miproj/mphil.acs.oct2012/mh693/code/hsst/bin/translate.py -v -r $lidx-$uidx $config

echo FINISHED: `date`
