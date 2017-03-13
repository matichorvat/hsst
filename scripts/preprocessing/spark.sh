#!/bin/bash

collect_spark() {
    if [ "$#" -lt 1 ]; then
	    echo "Usage: collect_spark input_dir [output_file] ['noremove']"
    else
	    spark_output_dir=${1%/}
	    output_file=${1%/}.txt
	    shift

        if [ "$#" -ge 1 ]; then
            output_file=$1
            shift
        fi

        zcat -f $spark_output_dir/* > $output_file

        #if [ "$#" -lt 1 ]; then
        #    rm -rf $spark_output_dir
        #fi
    fi
}
