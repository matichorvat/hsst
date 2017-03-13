import os
import logging
import argparse
import unicodedata

from pyspark import SparkContext, SparkConf, StorageLevel

from hsst.preprocessing.utility import clean_utf8, unescape, squeeze, filter_if_empty, filter_by_length, filter_by_fertility, filter_if_match, filter_by_language
from hsst.preprocessing.tokenizer import Tokenizer


PATH = '/home/miproj/mphil.acs.oct2012/mh693/code/hsst/tools/'
PERL_NORMALIZE_CMD = PATH + 'perl_scripts/normalize-punctuation.perl + {}'
AACHEN_TOKENIZER_CMD = PATH + 'corpusTools.28set2005/intel/applyTokenizer ' + PATH + 'corpusTools.28set2005/en.tokmap'
POST_AACHEN_TOKENIZER_CMD = PATH + 'scripts/postAachanTokeniser.sh'
POST_POST_AACHEN_TOKENIZER_CMD = PATH + 'perl_scripts/engtok-1.0/tokenizer.pl'
MOSES_TOKENIZER_CMD = PATH + 'perl_scripts/tokenize.perl'
MOSES_TRUECASER_CMD = PATH + 'recaser/truecase.perl'
MOSES_TRUECASER_DE_MODEL = PATH + 'recaser/truecase-model.eva.de'
AACHEN_TOKENIZER_TIMEOUT = 1.0


def read_parallel_file(filepath_1, filepath_2, sc, num_partitions=None):
    # Read the two parallel files in separately and zip them together
    with open(filepath_1) as f1, open(filepath_2) as f2:
        f1_lines = f1.read().decode('utf-8').strip().split('\n')
        f2_lines = f2.read().decode('utf-8').strip().split('\n')

        if num_partitions is not None:
            joined = sc.parallelize(zip(f1_lines, f2_lines), num_partitions)
        else:
            joined = sc.parallelize(zip(f1_lines, f2_lines))

    return joined


def read_mono_file(filepath, sc, num_partitions=None):
    if num_partitions is not None:
        return sc.textFile(filepath, num_partitions, use_unicode=True)
    else:
        return sc.textFile(filepath, use_unicode=True)


def load_mono_inputs(input_filepaths, sc, num_partitions):
    datasets = dict()
    for input_filepath in input_filepaths:
        dataset = read_mono_file(input_filepath, sc, num_partitions)
        datasets[input_filepath] = dataset
        dataset.persist(StorageLevel.MEMORY_AND_DISK)

    return datasets


def preprocess_de_mono(de_mono, clean=True, tok=True, tru=True):

    if clean:
        de_mono = clean_german(de_mono)

    if tok:
        de_mono = tokenize_german(de_mono)

    if tru:
        de_mono = truecase_german(de_mono)

    return de_mono


def preprocess_en_mono(en_mono, clean=True, tok=True):

    if clean:
        en_mono = clean_english(en_mono)

    en_mono = en_mono.filter(lambda x: len(x) < 2000)

    if tok:
        en_mono = tokenize_english(en_mono)

    return en_mono


def preprocess_en_de_parallel(en_de_parallel, expected_langs=None, untok=True):

    en_de_parallel = en_de_parallel.sample(False, 0.01, 1)

    en_parallel = en_de_parallel.map(lambda x: x[0])
    de_parallel = en_de_parallel.map(lambda x: x[1])

    # Clean each side separately. This does not remove any lines
    en_parallel_prep_clean = clean_english(en_parallel)
    en_parallel_prep_clean = en_parallel_prep_clean.zipWithIndex().map(lambda x: (x[1], x[0]))

    if untok:
        en_parallel_prep_clean.persist(StorageLevel.MEMORY_AND_DISK)

    # Combine English side with index keys and tokenize. Some lines may fail in tokenization, and will be removed
    en_parallel_prep = tokenize_english_indexed(en_parallel_prep_clean)
    # en_par_tok.values().saveAsTextFile(en_out)

    # Preprocess german side. No lines are removed here.
    de_parallel_prep = preprocess_de_mono(de_parallel)

    # Combine German side with index keys and join it with English side, removing lines that failed in English tokenization
    en_de_parallel_prep = en_parallel_prep.join(de_parallel_prep.zipWithIndex().map(lambda x: (x[1], x[0])))
    #.map(lambda x: (x[0], x[1][0]))

    filtered = filter_parallel_indexed(en_de_parallel_prep).sortByKey()

    if expected_langs is not None:
        filtered = filter_parallel_langs(filtered, expected_langs)

    filtered.persist(StorageLevel.MEMORY_AND_DISK)

    en_clean_joined = None
    if untok:
        filtered.persist(StorageLevel.MEMORY_AND_DISK)
        en_clean_joined = filtered.map(lambda x: (x[0], x[1][0])).join(en_parallel_prep_clean).map(lambda x: (x[0], x[1][1])).sortByKey().values()

    en_filtered = filtered.map(lambda x: (x[0], x[1][0])).values()
    de_filtered = filtered.map(lambda x: (x[0], x[1][1])).values()

    return en_filtered, de_filtered, en_clean_joined


def clean_german(source):
    return clean_source(source, 'de').map(lambda x: unicodedata.normalize('NFC', x))


def tokenize_german(source):
    return source.pipe('%s -l %s' % (MOSES_TOKENIZER_CMD, 'de')).map(squeeze)


def truecase_german(source):
    return source.pipe('%s --model %s' % (MOSES_TRUECASER_CMD, MOSES_TRUECASER_DE_MODEL))


def clean_english(source):
    return clean_source(source, 'en')


def tokenize_english(source):
    return source.mapPartitions(lambda partition: create_and_tokenize(AACHEN_TOKENIZER_CMD, partition)).pipe(POST_AACHEN_TOKENIZER_CMD).pipe(POST_POST_AACHEN_TOKENIZER_CMD).map(squeeze)


def tokenize_english_indexed(en_source):
    en_tok = en_source.mapPartitions(lambda partition: create_and_tokenize_indexed(AACHEN_TOKENIZER_CMD, partition))
    en_tok.persist(StorageLevel.MEMORY_AND_DISK)
    en_post_tok = en_tok.map(lambda x: x[1]).pipe(POST_AACHEN_TOKENIZER_CMD).pipe(POST_POST_AACHEN_TOKENIZER_CMD).map(squeeze)
    en_final_tok = en_tok.keys().zip(en_post_tok)
    return en_final_tok


def clean_source(source, lang):
    return source.map(clean_utf8).map(unescape).pipe(PERL_NORMALIZE_CMD.format(lang))


def create_and_tokenize(tokenize_cmd, partition):
    tokenizer = Tokenizer(tokenize_cmd, timeout=AACHEN_TOKENIZER_TIMEOUT)
    return filter(lambda x: x is not None, [tokenizer.tokenize(line) for line in partition])


def create_and_tokenize_indexed(tokenize_cmd, partition):
    tokenizer = Tokenizer(tokenize_cmd, timeout=AACHEN_TOKENIZER_TIMEOUT)
    return filter(lambda x: x[1] is not None, [(key, tokenizer.tokenize(en_line)) for key, en_line in partition])


def filter_mono(source, lang):
    return source.filter(lambda x: len(x.strip()) > 0 and x != '.' and x != '#' and len(x.strip().split(' ')) <= 100)  # and filter_by_language(x, lang))


def filter_parallel_indexed(source):
    return source.filter(lambda x: filter_if_empty(x[1])) \
                 .filter(lambda x: filter_if_match(x[1], '.')) \
                 .filter(lambda x: filter_if_match(x[1], '#')) \
                 .filter(lambda x: filter_by_length(x[1], 100)) \
                 .filter(lambda x: filter_by_fertility(x[1], 2.4))


def filter_parallel_langs(source, expected_langs):
    return source.filter(lambda x: filter_by_language(x[1][0], expected_langs[0]) and filter_by_language(x[1][1], expected_langs[1]))


if __name__ == '__main__':

    appName = 'WMT-15 Preprocessing'

    parser = argparse.ArgumentParser(description='Monolingual or parallel corpus preprocessing.')
    parser.add_argument('-l', '--log')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--partitions', type=int, default=50)
    parser.add_argument('-m', '--mono', choices=['en', 'de'], help='Preprocess a list of monolingual corpora in specified language, e.g. en or de.')
    parser.add_argument('-p', '--parallel', nargs=2, help='Preprocess a parallel corpus in specified language pair, e.g. en de')
    parser.add_argument('output_dir', help='Output directory.')
    parser.add_argument('inputs', nargs=argparse.REMAINDER, help='List of space-separated input files')

    args = parser.parse_args()

    # Set up logging
    if args.log:
        logging.basicConfig(filename=args.log, level=logging.DEBUG if args.verbose else logging.INFO)
    else:
        logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    num_partitions = args.partitions

    conf = SparkConf().setAppName(appName).set('spark.executor.extraJavaOptions', '-Dfile.encoding=UTF-8')
    sc = SparkContext(conf=conf)

    #if not conf.get('master').startswith('local'):
    #    outpath = 'file://' + args.output_dir
    #else:
    outpath = args.output_dir

    print outpath

    if args.mono:
        lang = args.mono

        # Load inputs to datasets dictionary
        datasets = load_mono_inputs(args.inputs, sc, num_partitions)

        for filepath, dataset in datasets.items():
            if lang == 'en':
                mono = preprocess_en_mono(dataset)

            elif lang == 'de':
                mono = preprocess_de_mono(dataset)

            mono = filter_mono(mono, lang)
            mono.saveAsTextFile(os.path.join(outpath, os.path.basename(filepath) + '.tok'))

    elif args.parallel and args.parallel == ['en', 'de']:
        en_de_parallel = read_parallel_file(args.inputs[0], args.inputs[1], sc, num_partitions)
        en_parallel, de_parallel, en_untok = preprocess_en_de_parallel(en_de_parallel, expected_langs=args.parallel)

        en_parallel.saveAsTextFile(os.path.join(outpath, os.path.basename(args.inputs[0]) + '.tok'))
        de_parallel.saveAsTextFile(os.path.join(outpath, os.path.basename(args.inputs[1]) + '.tok'))
        en_untok.saveAsTextFile(os.path.join(outpath, os.path.basename(args.inputs[0]) + '.untok'))
