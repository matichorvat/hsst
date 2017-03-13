import sys
import logging
import argparse

from pyspark import SparkContext, SparkConf, StorageLevel

from pydmrs.dmrs_preprocess import dmrs_preprocess


def load_wmap(filename):
    wmap = dict()

    with open(filename, 'rb') as fp:
        for line in fp:
            entry = line.strip().split('\t')

            assert len(entry) == 2

            wmap[entry[1].decode('utf-8')] = int(entry[0])

    return wmap


def read_files_old(sc, dmrs_filename, untok_filename, tok_filename, num_partitions, transfer_mt_prep=False):
    dmrs_list = dmrs_preprocess.read_file(dmrs_filename, format='dmrs')

    if not transfer_mt_prep:
        untok_list = dmrs_preprocess.read_file(untok_filename, format='untok')
        tok_list = dmrs_preprocess.read_file(tok_filename, format='tok')
    else:
        untok_list = [''] * len(dmrs_list)
        tok_list = [''] * len(dmrs_list)

    sys.stderr.write('Input lengths: %d %d %d\n' % (len(dmrs_list), len(untok_list), len(tok_list)))

    if not (len(dmrs_list) == len(untok_list) == len(tok_list)):
        raise Exception('Differing input lengths:  %d %d %d\n' % (len(dmrs_list), len(untok_list), len(tok_list)))

    joined = sc.parallelize(zip(dmrs_list, untok_list, tok_list), num_partitions)
    return joined


def read_files(sc, dmrs_filename, untok_filename, tok_filename, num_partitions):

    # Split with custom delimiter
    dmrs_dataset = sc.newAPIHadoopFile('file:' + dmrs_filename,
                                       'org.apache.hadoop.mapreduce.lib.input.TextInputFormat',
                                       'org.apache.hadoop.io.Text',
                                       'org.apache.hadoop.io.LongWritable',
                                       conf={'textinputformat.record.delimiter': '</dmrs>'}).repartition(num_partitions)

    # Reattach the delimiter to each record
    dmrs_dataset = dmrs_dataset.filter(lambda x: x[1].strip() != '').map(lambda x: x[1] + '</dmrs>')

    untok_dataset = sc.textFile(untok_filename, num_partitions, use_unicode=False)
    tok_dataset = sc.textFile(tok_filename, num_partitions, use_unicode=False)

    dmrs_num, untok_num, tok_num = dmrs_dataset.count(), untok_dataset.count(), tok_dataset.count()

    sys.stderr.write('Input lengths: %d %d %d\n' % (dmrs_num, untok_num, tok_num))

    if not (dmrs_num == untok_num == tok_num):
        raise Exception('Differing input lengths:  %d %d %d\n' % (dmrs_num, untok_num, tok_num))

    joined = dmrs_dataset.zip(untok_dataset).zip(tok_dataset).map(lambda x: (x[0][0], x[0][1], x[1]))

    return joined


def preprocess(dmrs_filename, untok_filename, tok_filename, output_file, sc, num_partitions, dmrs_preprocess_params, transfer_mt_prep=False):
    data = read_files_old(sc, dmrs_filename, untok_filename, tok_filename, num_partitions, transfer_mt_prep=transfer_mt_prep)
    prep = data.map(lambda x: dmrs_preprocess.process(x[0], x[1], x[2], **dmrs_preprocess_params))
    prep.saveAsTextFile(output_file)


if __name__ == "__main__":

    appName = 'DMRS preprocessing'

    parser = argparse.ArgumentParser(description='Preprocess DMRS graphs.')
    parser.add_argument('-l', '--log')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-p', '--partitions', type=int, default=100)
    parser.add_argument('--token_align', type=bool, default=True, help='Align tokens')
    parser.add_argument('--unaligned_align', type=bool, default=True, help='Use heuristic alignment to align unaligned tokens')
    parser.add_argument('--label', type=bool, default=True, help='Create node and edge labels')
    parser.add_argument('--handle_ltop', type=bool, default=True, help='Handle LTOP')
    parser.add_argument('--gpred_filter', help='Path to gpred filter file')
    parser.add_argument('--handle_unknown', type=bool, default=True, help='Handle unknown words (e.g. jumped/VBD).')
    parser.add_argument('--gpred_curb', type=int, default=3, help='Limit the length of gpred token alignment')
    parser.add_argument('--cycle_remove', type=bool, default=True, help='Remove cycles in the DMRS graph.')
    parser.add_argument('--map_node_tokens', default=None, help='Add tokens and token idx to nodes. Requires a word map file to be specified. Path to wmap file')
    parser.add_argument('--attach_tok', type=bool, default=True, help='Attach token string to DMRS')
    parser.add_argument('--attach_untok', type=bool, default=True, help='Attach untokenized string to DMRS')
    parser.add_argument('--realization', action='store_true', help='Turn on realization mode which does not use tokalign information in graph cycle removal.')
    parser.add_argument('--realization_sanity_check', action='store_true', help='Turn on sanity check mode for realization which strips all source sentence information from DMRS graphs.')
    parser.add_argument('--transfer_mt_prep', action='store_true', help='Preprocess DMRS obtained from transfer MT system.')
    parser.add_argument('dmrs_filepath')
    parser.add_argument('untok_filepath')
    parser.add_argument('tok_filepath')
    parser.add_argument('output_filepath')

    args = parser.parse_args()

    # Set up logging
    if args.log:
        logging.basicConfig(filename=args.log, level=logging.DEBUG if args.verbose else logging.INFO)
    else:
        logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    num_partitions = args.partitions

    if args.handle_unknown:
        import spacy
        lemmatizer = spacy.lemmatizer.Lemmatizer.from_package(spacy.util.get_package_by_name('en'))

    else:
        lemmatizer = None

    dmrs_preprocess_params = {'token_align_opt': args.token_align,
                              'unaligned_align_opt': args.unaligned_align,
                              'label_opt': args.label,
                              'handle_ltop_opt': args.handle_ltop,
                              'gpred_filter': dmrs_preprocess.filter_gpred.parse_gpred_filter_file(
                                  args.gpred_filter) if args.gpred_filter is not None else None,
                              'unknown_handle_lemmatizer': lemmatizer,
                              'gpred_curb_opt': args.gpred_curb,
                              'cycle_remove_opt': args.cycle_remove,
                              'map_node_tokens': load_wmap(args.map_node_tokens) if args.map_node_tokens else None,
                              'attach_tok': args.attach_tok,
                              'attach_untok': args.attach_untok,
                              'realization': args.realization,
                              'realization_sanity_check': args.realization_sanity_check,
                              'transfer_mt_prep': args.transfer_mt_prep
                              }

    conf = SparkConf().setAppName(appName).set('spark.executor.extraJavaOptions', '-Dfile.encoding=UTF-8')
    sc = SparkContext(conf=conf)

    preprocess(args.dmrs_filepath, args.untok_filepath, args.tok_filepath, args.output_filepath, sc, num_partitions, dmrs_preprocess_params, transfer_mt_prep=args.transfer_mt_prep)
