import json
import logging
import argparse
import xml.etree.ElementTree as xml

from pyspark import SparkContext, SparkConf, StorageLevel

from hsst.utility import representation
from hsst.preprocessing.filtering import SourceTokLength, SourceGraphSizeFilter
from hsst.rulextraction.rulextraction_interface import string_dataset_translation_rulextraction, string_dataset_realization_rulextraction


def load_inputs(input_filepaths, sc, num_partitions):
    datasets = list()
    for input_filepath in input_filepaths:
        dataset = read_file(sc, 'file:' + input_filepath, num_partitions / len(input_filepaths) if num_partitions / len(input_filepaths) > 5 else 5)
        datasets.append(dataset)

    joint_dataset = sc.union(datasets)

    return joint_dataset


def read_file(sc, filename, num_partitions):
    # Split with custom delimiter
    logging.info('Loading dataset with filename {}'.format(filename))

    dataset = sc.newAPIHadoopFile(
        filename,
        'org.apache.hadoop.mapreduce.lib.input.TextInputFormat',
        'org.apache.hadoop.io.Text',
        'org.apache.hadoop.io.LongWritable',
        conf={'textinputformat.record.delimiter': '</sentence>'}
    ) #.repartition(num_partitions)

    # Reattach the delimiter to each record
    return dataset.filter(lambda x: x[1].strip() != '').map(lambda x: x[1] + '</sentence>')


def filter_data(data, filtering):
    return data.filter(lambda x: filter_sentence(x, filtering))


def filter_sentence(sentence, filtering, format='xml'):
    try:
        sentence_collection = representation.loads(sentence, format=format)
    except xml.ParseError:
        raise Exception('Error parsing sentence XML for id %d' % id)

    sentence = next(iter(sentence_collection))

    return filtering.filter(sentence)


def sample_data(data, sample_n):
    data_size = data.count()
    fraction = sample_n / float(data_size)

    if fraction >= 1.0:
        return data

    return data.sample(False, fraction)


def rule_to_json(rule):
    src_nodes = [(str(x.node_id), x.label) for x in sorted(rule.source_side.nodes, key=lambda x: x.node_id)]
    src_edges = [(str(x.from_node.node_id), str(x.to_node.node_id), x.label) for x in sorted(rule.source_side.edges, key=lambda x: x.edge_id)]
    src = (tuple(src_nodes), tuple(src_edges))
    trg = tuple(rule.target_side)
    rule_repr = (src, trg)

    return None, json.dumps(rule_repr)


def rule_extract(data, rule_extract_func, idx=True, filtering=None, timeout=None, subgraph_df_limit=20):
    rules = data.flatMap(lambda x: rule_extract_func(
        x,
        idx=idx,
        filtering=filtering,
        timeout=timeout,
        subgraph_df_limit=subgraph_df_limit
    ))

    return rules


def process(data, ruleset_filename, num_partitions, rule_extract_func, idx=True, filtering=None, timeout=None, sample=None, subgraph_df_limit=20):

    if filtering is not None:
        data = filter_data(data, filtering)

    data.persist(StorageLevel.MEMORY_ONLY)
    if sample:
        data = sample_data(data, sample)

    data = data.repartition(num_partitions)

    # Rule extract
    rules = rule_extract(
        data,
        rule_extract_func,
        idx=idx,
        timeout=timeout,
        subgraph_df_limit=subgraph_df_limit
    )

    # Transform rule objects to json string
    rules = rules.map(rule_to_json)

    # Save ruleset to HDFS
    rules.saveAsSequenceFile(ruleset_filename)


if __name__ == "__main__":

    appName = 'Rule Extraction'

    parser = argparse.ArgumentParser(description='Do rulextraction with Spark and save ruleset.')
    parser.add_argument('-l', '--log')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-p', '--partitions', type=int, default=100)
    parser.add_argument('-op', '--operation', default='translation', choices=['translation', 'realization'])
    parser.add_argument('-m', '--mode', default='sgtt', choices=['sgtt', 'sttg', 'sgtg', 'sttt'])
    parser.add_argument('-dfl', '--subgraph_df_limit', type=int, default=100)
    parser.add_argument('-t', '--timeout', type=int, default=0)
    parser.add_argument('-f', '--filter', choices=['token_num', 'graph_size'])
    parser.add_argument('-fmax', '--filtermax', type=int, default=0)
    parser.add_argument('-fmin', '--filtermin', type=int, default=0)
    parser.add_argument('-s', '--sample', type=int, default=0)
    parser.add_argument('--idx', type=bool, default=True)
    parser.add_argument('output_filepath')
    parser.add_argument('inputs', nargs=argparse.REMAINDER, help='List of space-separated input DS files')

    args = parser.parse_args()

    # Set up logging
    if args.log:
        logging.basicConfig(filename=args.log, level=logging.DEBUG if args.verbose else logging.INFO)
    else:
        logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    num_partitions = args.partitions

    conf = SparkConf().setAppName(appName).set('spark.executor.extraJavaOptions', '-Dfile.encoding=UTF-8').set(
        'spark.network.timeout', 600)
    sc = SparkContext(conf=conf)

    if args.mode == 'sgtt':
        if args.operation == 'translation':
            rule_extract_func = string_dataset_translation_rulextraction
        elif args.operation == 'realization':
            rule_extract_func = string_dataset_realization_rulextraction
        else:
            logging.exception('Unknown operation mode %s.' % (args.op))
            raise Exception('Unknown operation mode %s.' % (args.op))

        if args.filter == 'token_num':
            filtering = SourceTokLength(max_len=args.filtermax, min_len=args.filtermin)
        elif args.filter == 'graph_size':
            filtering = SourceGraphSizeFilter(max_nodes=args.filtermax, min_nodes=args.filtermin)
        else:
            filtering = None

    else:
        logging.exception('Modes other than sgtt not implemented.')
        raise NotImplementedError('Modes other than sgtt not implemented.')

    joint_dataset = load_inputs(args.inputs, sc, num_partitions)
    grammar_out = args.output_filepath if args.output_filepath.startswith('hdfs:') else 'file://' + args.output_filepath

    process(
        joint_dataset,
        grammar_out,
        num_partitions,
        rule_extract_func,
        idx=args.idx,
        filtering=filtering,
        timeout=args.timeout,
        sample=args.sample,
        subgraph_df_limit=args.subgraph_df_limit
    )
