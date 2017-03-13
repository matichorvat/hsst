#!/bin/bash

function realize_1best() {
    > $2
    while read mrs;
    do
        output=`echo $mrs | $ACEDIR/ace -g $erg_grammar -e -n 1 2>>logs/$3.log`
        echo $output >> $2
    done < $1
}

function realize_nbest() {
    mkdir -p $3
    linenum=1
    while read mrs;
    do
        echo $mrs | $ACEDIR/ace -g $erg_grammar -e -n $2 2>>logs/$4.log > $3/$linenum.txt
        linenum=$((linenum+1))
    done < $1
}

function realize_1best_pp() {
    > $2
    while read mrs;
    do
        output=`echo $mrs | /home/miproj/mphil.acs.oct2012/mh693/code/hsst/scripts/ace_pp_realization.py "$ACEDIR/ace -g $erg_grammar_pp -e -n 1 2>>logs/$3.log"`
        echo $output >> $2
    done < $1
}

function realize_nbest_pp() {
    mkdir -p $3
    linenum=1
    while read mrs;
    do
        echo $mrs | /home/miproj/mphil.acs.oct2012/mh693/code/hsst/scripts/ace_pp_realization.py "$ACEDIR/ace -g $erg_grammar_pp -e -n $2 2>>logs/$4.log" > $3/$linenum.txt
        linenum=$((linenum+1))
    done < $1
}

function combined_system() {
    erg_file=$1
    hssr_file=$2

    linenum=1
    while read line; do
        if [ "$line" == "" ]; then
            sed "${linenum}q;d" $hssr_file
        else
            echo $line
        fi
        linenum=$((linenum+1))
    done < $erg_file
}

function realize_1best_mt() {
    log="logs/$3.log"
    > $2
    > $log
    while read mrs;
    do
        output=`echo $mrs | $ACEDIR/ace -g $erg_grammar -e -1 --disable-subsumption-test 2>>$log`
        echo $output >> $2
    done < $1
}

function realize_from_nbest_mt() {
    inputdir=$1
    num=$2
    outputfile=$3
    mrsoutfile=$4
    log="logs/$5.log"

    > $outputfile
    > $mrsoutfile
    > $log

    for i in $(seq 1 $num); do
        echo $i
        filename=$inputdir/$i.transf
        while read mrs;
        do
            output=`echo $mrs | $ACEDIR/ace -g $erg_grammar -1e --disable-subsumption-test 2>>$log`
            if [[ ! -z "${output// }" ]]; then
                echo $mrs >> $mrsoutfile
                break
            fi
        done < $filename
        echo $output >> $outputfile
        if [[ -z "${output// }" ]]; then
            echo 'None' >> $mrsoutfile
        fi
    done
}

function realize_1best_pp_mt() {
    log="logs/$3.log"
    > $2
    > $log
    while read mrs;
    do
        output=`echo $mrs | /home/miproj/mphil.acs.oct2012/mh693/code/hsst/scripts/ace_pp_transfer_realization.py "$ACEDIR/ace -g $erg_grammar_pp -e -1 --disable-subsumption-test 2>>$log"`
        # echo $mrs | /home/miproj/mphil.acs.oct2012/mh693/code/hsst/scripts/ace_pp_transfer_realization.py "$ACEDIR/ace -g $erg_grammar_pp -e -1 --disable-subsumption-test"
        echo $output >> $2
    done < $1
}

function realize_from_nbest_pp_mt() {
    inputdir=$1
    num=$2
    outputfile=$3
    mrsoutfile=$4
    log="logs/$5.log"

    > $outputfile
    > $mrsoutfile
    > $log

    for i in $(seq 1 $num); do
        echo $i 'pp'
        filename=$inputdir/$i.transf
        while read mrs;
        do
            output=`echo $mrs | /home/miproj/mphil.acs.oct2012/mh693/code/hsst/scripts/ace_pp_transfer_realization.py "$ACEDIR/ace -g $erg_grammar -1e --disable-subsumption-test 2>>$log"`
            if [[ ! -z "${output// }" ]]; then
                echo $mrs >> $mrsoutfile
                break
            fi
        done < $filename
        echo $output >> $outputfile
        if [[ -z "${output// }" ]]; then
            echo 'None' >> $mrsoutfile
        fi
    done
}

function BLEUeval() {
    /home/miproj/mphil.acs.oct2012/mh693/code/hsst/scripts/scoring/score_MThyp.sh -h $1 -ref $2 -src $3 -s $4 -srclang en -trglang en -odir $5 | tee $5/$4.log
}

function HSSR_nbest() {
    # Setup directories
    rm -rf $4
    mkdir -p tmp_dir $4

    # Save n-best hypotheses to temporary directory
    /home/miproj/mphil.acs.oct2012/mh693/installs/hifst/hifst.2016-05-09/ucam-smt/bin/printstrings.sta.O2.bin -u -m /data/mifs_scratch/mh693/wmt15/0132-wmt15-en-de/wmaps/wmt15.en-de.en.unmap -n $2 --input=$1/?.fst.gz --output=tmp_dir/?.txt --range=1:$3

    # Post-process n-best hypotheses
    for i in $(seq 1 $3); do
        cat tmp_dir/$i.txt | awk '{$1=$(NF)=""; print $0}' | /home/miproj/mphil.acs.oct2012/mh693/code/hsst/scripts/scoring/detokenizer.perl -l en | /home/miproj/mphil.acs.oct2012/mh693/code/hsst/scripts/scoring/detruecase.perl >> $4/$i.txt
    done

    rm -rf tmp_dir
}

function is_sentence_nbest_parseable() {
    linenum=1
    while read line;
    do
        if [ "$linenum" -gt $2 ]; then
            break
        fi

        parse_log=`echo $line | $ACEDIR/ace -g $erg_grammar -1TR 2>&1`
        echo $parse_log | grep -q '1 / 1 sentences'
        parsed=$?

        if [ "$parsed" -eq 0 ]; then
            return 0
        fi

        linenum=$((linenum+1))
    done < $1

    return 1
}

function dataset_nbest_parseable() {
    num_parsed=0
    num_files=0
    for filename in $1/*.txt; do
        # Returns 0 if parsed, 1 if not parsed
        is_sentence_nbest_parseable $filename $2

        parsed=$?
        if [ "$parsed" -eq 0 ]; then
            num_parsed=$((num_parsed+1))
        fi

        num_files=$((num_files+1))
    done

    parsed_percentage=`bc -l <<< "$num_parsed / $num_files"`

    printf "# Parsed: %0.3f (%d/%d) for %s\n" $parsed_percentage $num_parsed $num_files $1
}

function all_nbest_parseable() {
    for dir in $1/*; do
        hssr_nbest=$(basename $dir | cut -f1 -d_)
        dataset_nbest_parseable $dir $hssr_nbest
    done
}


function num_realized() {
    total=$2
    num_empty=`grep -cvP '\S' $1`
    num_realized=$((total - num_empty))
    echo `bc <<< "scale=2; $num_realized/$total"` $num_realized
}
