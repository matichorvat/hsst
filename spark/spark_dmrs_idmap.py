import os
import sys
import logging
import argparse
from operator import add
import xml.etree.ElementTree as xml
from xml.etree.ElementTree import ParseError

from pyspark import SparkContext, SparkConf, StorageLevel

from pydmrs.dmrs_idmap import dmrs_idmap
from pydmrs.dmrs_idmap.vocab import SourceGraphVocab
from pydmrs.dmrs_idmap.wmap import SourceGraphWMAP


def empty(dmrs_xml):
    for _ in dmrs_xml:
        return False

    return True


def read_file(sc, dmrs_filename, num_partitions):
    dmrs_list = dmrs_idmap.read_file(dmrs_filename)
    joined = sc.parallelize(dmrs_list, num_partitions)
    return joined


def extract_vocab(dmrs_list):
    vocab_extractor = SourceGraphVocab()

    for dmrs in dmrs_list:
        parser = xml.XMLParser(encoding='utf-8')

        try:
            dmrs_xml = xml.fromstring(dmrs.encode('utf-8'), parser=parser)

        except ParseError:
            sys.stderr.write(dmrs + "\n")
            raise

        vocab_extractor.extract_sentence(dmrs_xml)

    return vocab_extractor.get_freq().most_common()


def create_wmap(vocab, existing_wmap=None):

    if existing_wmap is None:
        wmap = SourceGraphWMAP()
    else:
        wmap = SourceGraphWMAP(existing_wmap)

    for key, _ in vocab.toLocalIterator():
        wmap.get_or_add_value(key)

    return wmap


def idmap(dmrs, wmap):
    parser = xml.XMLParser(encoding='utf-8')

    try:
        dmrs_xml = xml.fromstring(dmrs.encode('utf-8'), parser=parser)
    except ParseError:
        sys.stderr.write(dmrs + "\n")
        raise

    if empty(dmrs_xml):
        return dmrs
    else:
        wdmrs = wmap.wmap_sentence(dmrs_xml)
        return xml.tostring(wdmrs, encoding='utf-8')


def load_inputs(input_filepaths, sc, num_partitions):
    datasets = dict()
    for input_filepath in input_filepaths:
        dataset = read_file(sc, input_filepath, num_partitions)
        datasets[input_filepath] = dataset
        dataset.persist(StorageLevel.MEMORY_AND_DISK)

    return datasets


def extract_joint_vocab(datasets, sc):
    joint_dataset = sc.union(datasets.values())
    vocab = joint_dataset.mapPartitions(extract_vocab).reduceByKey(add).sortBy(lambda x: x[1], ascending=False)
    vocab.persist(StorageLevel.MEMORY_AND_DISK)

    return vocab


def idmap_dataset(dataset, output_filepath, wmap, sc, num_partitions):
    data_idx = dataset.map(lambda x: idmap(x, wmap.value))
    data_idx.saveAsTextFile(output_filepath)


if __name__ == "__main__":

    appName = 'DMRS idmap'

    parser = argparse.ArgumentParser(description='Map DMRS graph labels to ids.')
    parser.add_argument('-l', '--log')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-p', '--partitions', type=int, default=100)
    parser.add_argument('-s', '--suffix', default='idx', help='File suffix of id mapped outputs')
    parser.add_argument('-o', '--output_dir', help='Directory to output id mapped inputs to')
    parser.add_argument('-w', '--wmap_in', help='Existing WMAP file')
    parser.add_argument('--vocab_out', help='If specified, joint vocab output filepath')
    parser.add_argument('wmap_out', help='Word map output file')
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

    # Load inputs to datasets dictionary
    datasets = load_inputs(args.inputs, sc, num_partitions)

    # Extract joint vocabulary
    vocab = extract_joint_vocab(datasets, sc)

    # Create WMAP
    word_map = create_wmap(vocab, existing_wmap=args.wmap_in)

    # Save WMAP to disk
    with open(args.wmap_out, 'wb') as out:
        out.write(str(word_map))

    if args.vocab_out:
        vocab.map(lambda x: '%s\t%d' % (x[0], x[1])).saveAsTextFile(args.vocab_out)

    word_map_BC = sc.broadcast(word_map)

    # ID map datasets
    for input_filepath, dataset in datasets.items():
        input_filename = os.path.basename(input_filepath)
        input_basename = os.path.splitext(input_filename)[0]
        output_filename = input_basename + '.' + args.suffix

        if args.output_dir:
            output_filepath = os.path.join(args.output_dir, output_filename)
        else:
            input_dirname = os.path.dirname(input_filepath)
            output_filepath = os.path.join(input_dirname, output_filename)

        idmap_dataset(dataset, output_filepath, word_map_BC, sc, num_partitions)


