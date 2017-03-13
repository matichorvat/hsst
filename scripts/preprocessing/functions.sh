#!/bin/bash

entokenize() {
    export SOFTMT_SOURCE_DIR=/home/blue1/ad465/softmt/source
    tools="scripts/corpusTools.28set2005"
    export tools=${tools}
    function tokenize {
	${tools}/intel/applyTokenizer ${tools}/$1.tokmap
    }

    input=$1
    output=$2
    zcat -f $input | scripts/clean-utf8.pl | tokenize en > data1.en
    cat data1.en | sed -e 's/_NUMBER { \([^}]\+\) }/\1/g' -e 's/_NUMNBER { \([^}]\+\) }/\1/g'\
	-e 's/_WWW[ ]{ \([^}]\+\) }/\1/g' -e 's/_WWW{\([^}]\+\)}/\1/g' -e 's/{ \([^}]\+\) }/\1/g' |\
	iconv -c -f utf-8 -t iso-8859-1 > data2.en
    cat data2.en | $SOFTMT_SOURCE_DIR/mt-tokenise/bin/mt-tokenise.bash eng |\
	sed -e 's/ \+/ /g' -e 's/ $//g' -e 's/^ //g' -e "s: ' s : 's :g" | iconv -t utf-8 -f iso-8859-1 | gzip > $output
    rm -f data1.en data2.en
}

ensure_encoding() {
    export LANG=en_GB.UTF-8
    export LC_CTYPE="en_GB.UTF-8"
    export LC_NUMERIC="en_GB.UTF-8"
    export LC_TIME="en_GB.UTF-8"
    export LC_COLLATE="en_GB.UTF-8"
    export LC_MONETARY="en_GB.UTF-8"
    export LC_MESSAGES="en_GB.UTF-8"
    export LC_PAPER="en_GB.UTF-8"
    export LC_NAME="en_GB.UTF-8"
    export LC_ADDRESS="en_GB.UTF-8"
    export LC_TELEPHONE="en_GB.UTF-8"
    export LC_MEASUREMENT="en_GB.UTF-8"
    export LC_IDENTIFICATION="en_GB.UTF-8"
    export LC_ALL=
}

word_to_id_map_full() {
    ensure_encoding

    scripts_dir=$1
    shift
    input_file=$1
    shift
    basename=$1
    shift
    output_file=$1
    shift
    wmap_dir=$1
    shift
    wmap_version=$1
    shift
    wmap_old=$1

    maptext=$scripts_dir/maptext_v2
    addnew2wmap=$scripts_dir/addnew2wmap
    generate_vocab=$scripts_dir/generate_vcb.pl

    mkdir -p $wmap_dir

    # Generate vocabulary
    zcat -f $input_file | awk '{printf("%s\n", tolower($0));}' | $generate_vocab > $wmap_dir/$basename.vcb

    # If an existing wmap is specified, extend it with the new vocabulary, otherwise create a new wmap
    if [ "$wmap_old" != "" ]; then
        zcat -f $input_file | awk '{printf("%s\n", tolower($0));}' | $addnew2wmap $wmap_old > $wmap_dir/$basename.wmap.v$wmap_version
        ln -sf $basename.wmap.v$wmap_version $wmap_dir/$basename.wmap
    else
        awk 'BEGIN {printf("0\t<epsilon>\n"); printf("1\t<s>\n"); printf("2\t</s>\n");}' > $wmap_dir/$basename.wmap.v$wmap_version
        cat $wmap_dir/$basename.vcb | awk '{printf("%d\t%s\n", NR+2, $1);}' >> $wmap_dir/$basename.wmap.v$wmap_version
        ln -sf $basename.wmap.v$wmap_version $wmap_dir/$basename.wmap
    fi

    # Map words to ids based on the new or extended wmap
    zcat -f $input_file | awk '{printf("%s\n", tolower($0));}' | $maptext $wmap_dir/$basename.wmap > $output_file #| gzip 
}

start_wmap() {
    ensure_encoding

    scripts_dir=$1
    shift
    output_file=$1
    shift
    lower_case=$1
    shift
    corpora="$*"

    generate_vocab=$scripts_dir/generate_vcb.pl
    addnew2wmap=$scripts_dir/addnew2wmap
    
    # Begin with standard symbols
    awk 'BEGIN {printf("0\t<epsilon>\n"); printf("1\t<s>\n"); printf("2\t</s>\n");}' > $output_file

    # Compute the vocabulary of all corpora and rewrite it as a word map
    zcat -f $corpora | ( [[ "${lower_case}" != "true" ]] && cat || awk '{printf("%s\n", tolower($0));}') | $generate_vocab | awk '{printf("%d\t%s\n", NR+2, $1);}' >> $output_file
}

extend_wmap() {
    ensure_encoding

    scripts_dir=$1
    shift
    wmap_old=$1
    shift
    output_file=$1
    shift
    corpora="$*"

    addnew2wmap=$scripts_dir/addnew2wmap

    zcat -f $corpora | awk '{printf("%s\n", tolower($0));}' | $addnew2wmap $wmap_old > $output_file
}

extend_wmap_nolc() {
    ensure_encoding

    scripts_dir=$1
    shift
    wmap_old=$1
    shift
    output_file=$1
    shift
    corpora="$*"

    addnew2wmap=$scripts_dir/addnew2wmap

    zcat -f $corpora | $addnew2wmap $wmap_old > $output_file
}


word_to_id_map() {
    ensure_encoding

    scripts_dir=$1
    shift
    input_file=$1
    shift
    output_file=$1    
    shift
    vocab_file=$1
    shift
    wmap=$1
    shift
    lower_case=$1

    maptext=$scripts_dir/maptext_v2
    generate_vocab=$scripts_dir/generate_vcb.pl

    # Generate vocabulary                                                                                                                                                                                                                    
    zcat -f $input_file | ( [[ "${lower_case}" != "true" ]] && cat || awk '{printf("%s\n", tolower($0));}' ) | $generate_vocab > $vocab_file

    # Map words to ids based on the new or extended wmap
    zcat -f $input_file | ( [[ "${lower_case}" != "true" ]] && cat || awk '{printf("%s\n", tolower($0));}' ) | $maptext $wmap > $output_file #| gzip                                                                                                                     
}


maptoidsnew() {
    ensure_encoding

    maptext=scripts/maptext_v2
    addnew2wmap=scripts/addnew2wmap

    
    input=$1
    odir=$2
    outdir=$3
    lang=$4
    corpus=$5
    newwmap=$6
    mkdir -p $odir

    generate_vocab="scripts/generate_vcb.pl"
    zcat -f $input | awk '{printf("%s\n", tolower($0));}' | $generate_vocab > $odir/$corpus.$lang.vcb

    if [ "$newwmap" == "" ]; then
        ### we recycle a wmap
	zcat -f $input | awk '{printf("%s\n", tolower($0));}' | $addnew2wmap $WMAP > $odir/$corpus.$lang.wmap.v$V
	#gzip $WMAP.v$((V-1))               # save previous version of the wmap
	#chmod 444 $odir/$corpus.$lang.wmap.v$((V-1)).gz
	ln -sf $corpus.$lang.wmap.v$V $odir/$corpus.$lang.wmap
    else
	### new wmap
	awk 'BEGIN {printf("0\t<epsilon>\n"); printf("1\t<s>\n"); printf("2\t</s>\n");}' > $odir/$corpus.$lang.wmap.v$V 
	cat $odir/$corpus.$lang.vcb | awk '{printf("%d\t%s\n", NR+2, $1);}' >> $odir/$corpus.$lang.wmap.v$V
	ln -sf $corpus.$lang.wmap.v$V $odir/$corpus.$lang.wmap
    fi

    zcat -f $input | awk '{printf("%s\n", tolower($0));}' | $maptext $odir/$corpus.$lang.wmap | gzip > $outdir/$corpus.$lang.tok.idx.gz
}


maptoidsdir() {
    ensure_encoding

    maptext=scripts/maptext_v2
    addnew2wmap=scripts/addnew2wmap

    input=$1
    odir=$2
    outdir=$3
    lang=$4
    corpus=$5
    newwmap=$6
    mkdir -p $odir

    generate_vocab="scripts/generate_vcb.pl"
    zcat -f $input/*.gz | awk '{printf("%s\n", tolower($0));}' | $generate_vocab > $odir/$corpus.$lang.vcb

    if [ "$newwmap" == "" ]; then
        ### we recycle a wmap                                                                                                                                                                                                                 
        zcat -f $input/*.gz | awk '{printf("%s\n", tolower($0));}' | $addnew2wmap $WMAP > $odir/$corpus.$lang.wmap.v$V
        #gzip $WMAP.v$((V-1))               # save previous version of the wmap                                                                                                                                                               
        #chmod 444 $odir/$corpus.$lang.wmap.v$((V-1)).gz                                                                                                                                                                                      
        ln -sf $corpus.$lang.wmap.v$V $odir/$corpus.$lang.wmap
    else
        ### new wmap                                                                                                                                                                                                                          
        awk 'BEGIN {printf("0\t<epsilon>\n"); printf("1\t<s>\n"); printf("2\t</s>\n");}' > $odir/$corpus.$lang.wmap.v$V
        cat $odir/$corpus.$lang.vcb | awk '{printf("%d\t%s\n", NR+2, $1);}' >> $odir/$corpus.$lang.wmap.v$V
        ln -sf $corpus.$lang.wmap.v$V $odir/$corpus.$lang.wmap
    fi

    for file in $input/*.gz
    do
	    file_id=$(basename $file | cut -d '.' -f 1 )
	    zcat -f $file | awk '{printf("%s\n", tolower($0));}' | $maptext $odir/$corpus.$lang.wmap | gzip > $outdir/$file_id.tok.idx.gz
    done
}

createVocabs() {
    src=$1
    trg=$2
    odir=$3
    generate_vocab=/home/miproj/mphil.acs.oct2012/mh693/code/hsst/scripts/preprocessing/generate_vcb.pl
    
    mkdir -p $odir

    # generate source and target vocabs
    zcat -f $src | $generate_vocab > $odir/`basename $src`.vcb
    zcat -f $trg | $generate_vocab > $odir/`basename $trg`.vcb
}

get_oovs() {
    input=$1
    vocab=$2
    output=$3
    zcat -f $input | perl /home/miproj/mphil.acs.oct2012/mh693/code/hsst/scripts/preprocessing/turn_oovs_into_ascii.pl $vocab > $output
}

add_oovs_to_wmap() {
    input=$1
    trgwmap=$2
    version=$3
    addnew2wmap=/home/miproj/mphil.acs.oct2012/mh693/code/hsst/scripts/preprocessing/addnew2wmap

    # Create an additional wordmap with OOVs included
    zcat -f $input | sed 's:^[0-9]* ::' | $addnew2wmap $trgwmap > $trgwmap.v$version+oovs   
}

idx-oovsets() {
    input=$1
    srcwmap=$2
    trgwmap=$3
    output=$4
    maptext=/home/miproj/mphil.acs.oct2012/mh693/code/hsst/scripts/preprocessing/maptext_v2
   
    # Map in src and trg wmaps:                                                                                                                                                                                                                     
    a=`basename $input`
    cat $input | awk '{print $1;}' > $a.src.tmp-lnums
    cat $input | sed 's:^[0-9]* ::' | $maptext $srcwmap > $a.src.tmp-src
    cat $input | sed 's:^[0-9]* ::' | $maptext $trgwmap > $a.src.tmp-trg
    paste -d '#' $a.src.tmp-lnums $a.src.tmp-src $a.src.tmp-trg | sed -e 's:#:\: :' -e 's:#: # :g' > $output
    rm -f $a.src.tmp-*
}

get_ascii() {
    input=$1
    vocab=$2
    output=$3
    turn_oovs_into_ascii=scripts/turn_oovs_into_ascii.pl
    zcat -f $input | awk '{printf("%s\n", tolower($0));}' | $turn_oovs_into_ascii $vocab > $output
}

idx-asciisets() {
    input=$1
    srcwmap=$2
    trgwmap=$3
    trgwmapversion=$4
    output=$5
    maptext=scripts/maptext_v2
    addnew2wmap=scripts/addnew2wmap
    zcat -f $input | sed 's:^[0-9]* ::' | $addnew2wmap $trgwmap > $trgwmap.v$trgwmapversion
    gzip $trgwmap.v$((trgwmapversion-1))               # save previous version of the wmap
    chmod 444 $trgwmap.v$((trgwmapversion-1)).gz
    ln -sf $trgwmap.v$trgwmapversion $trgwmap
    # Now map in src and trg wmaps:
    a=`basename $input`
    cat $input | awk '{print $1;}' > $a.src.tmp-lnums
    cat $input | sed 's:^[0-9]* ::' | $maptext $srcwmap > $a.src.tmp-src
    cat $input | sed 's:^[0-9]* ::' | $maptext $trgwmap > $a.src.tmp-trg
    paste -d '#' $a.src.tmp-lnums $a.src.tmp-src $a.src.tmp-trg | sed -e 's:#:\: :' -e 's:#: # :g' > $output
    rm -f $a.src.tmp-*
}

lowercasedir() {

    dir=$1

    for file in $dir/*
    do
        cat $file | awk '{printf("%s\n", tolower($0));}' > $file.lcase
        mv $file.lcase $file
    done

}

setsplit() {
    data_prep=$1
    shift
    dir=$1
    shift
    files="$*"

    mkdir -p $dir $dir/ds.idx $dir/tok $dir/idx

    for file_id in $files
    do
	mv $data_prep/ds.idx/$file_id.ds.idx.gz $dir/ds.idx/
	mv $data_prep/tok/$file_id.tok.gz $dir/tok/
	mv $data_prep/idx/$file_id.tok.idx.gz $dir/idx/
    done

}

create_dmrs_vocab() {
    dir=$1
    out=$2
    zcat $dir/* | grep -oPh '(?<=label=")[^ ]+(?=" )' | sort | uniq -c | sort -nr | awk ' { t = $1; $1 = $2; $2 = t; print; } ' | tr ' ' '\t' > $out
}

merge_vocabs() {
    dir=$1
    out=$2
    cat $dir/* | awk -F '\t' '{array[$1]+=int($2)} END { for (i in array) {print i"\t"int(array[i])}}' | sort -nr -k 2 > $out
}

create_dmrs_wmap() {
    vocab=$1
    out=$2
    id=0
    cut -d$'\t' -f 1 $vocab | while read line; do echo -e "$id\t$line"; id=$[$id + 1]; done > $out
}

carg_to_wmap() {
    export LANG=en_GB.UTF-8
    export LC_CTYPE="en_GB.UTF-8"
    export LC_NUMERIC="en_GB.UTF-8"
    export LC_TIME="en_GB.UTF-8"
    export LC_COLLATE="en_GB.UTF-8"
    export LC_MONETARY="en_GB.UTF-8"
    export LC_MESSAGES="en_GB.UTF-8"
    export LC_PAPER="en_GB.UTF-8"
    export LC_NAME="en_GB.UTF-8"
    export LC_ADDRESS="en_GB.UTF-8"
    export LC_TELEPHONE="en_GB.UTF-8"
    export LC_MEASUREMENT="en_GB.UTF-8"
    export LC_IDENTIFICATION="en_GB.UTF-8"
    export LC_ALL=

    addnew2wmap=scripts/addnew2wmap

    input=$1
    wmap=$2
    ev=$3

    zcat -f $input | awk '{printf("%s\n", tolower($0));}' | $addnew2wmap $wmap > $wmap.v$ev
    ln -sf $wmap.v$ev $wmap
}

apply_to_all_in_dir() {
    target_dir=$1
    command=$2
    
    current_dir=$(pwd)
    cd $target_dir
    ls | xargs -n 1000 $command
    cd $current_dir  
}

getArray() {
    i=0
    while read line # Read a line                                                                                                                                                                                             
    do
        array[i]=$line # Put it into the array
	i=$(($i + 1))
    done < $1
}
