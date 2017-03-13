#!/bin/bash
#$ -S /bin/bash

hostname
echo "START: "`date`

if [[ $# != 7 ]]; then
    echo "Args: [input] [parser] [grammar] [number of parses] [output] [log] [numparts]" > /dev/stderr
    exit
fi

source /home/miproj/mphil.acs.oct2012/mh693/code/hsst/scripts/preprocessing/functions.sh
ensure_encoding

p=$SGE_TASK_ID

input=$1
shift
parser=$1
shift
grammar=$1
shift
nparses=$1
shift
output=$1
shift
log=$1
shift
numparts=$1
shift

tsttot=`wc -l $input | cut -d " " -f 1`

### gather tst ids to process
tpartsize=$[$tsttot / $numparts]
rest=$[$tsttot % $numparts]
if [[ $rest -gt 0 && $p -le $rest ]]; then
    start=$[($tpartsize + 1) * ($p - 1) + 1 ]
    end=$[$start + $tpartsize]
else
    start=$[$tpartsize * ($p - 1) + 1 + $rest]
    end=$[$start + $tpartsize - 1]
fi

echo "Lines $lidx to $uidx"

# TNT needs to be in PATH

# sed -e 's/${/ ${/g' is there to fix a bug in ACE which makes it crash with an error when encountering '${' with a non-whitespace character in front
set -x
sed -n $start,${end}p $input | sed -e 's/${/ ${/g' | $parser -g $grammar -T -n $nparses --max-chart-megabytes 7000 --max-unpack-megabytes 8000 --timeout 15 -r "root_informal root_inffrag" --tnt-model=/home/miproj/mphil.acs.oct2012/mh693/installs/tnt/models/wsj.tnt 1> $output.$start.$end.parse 2> $log.$start.$end.log

echo "END: "`date`
