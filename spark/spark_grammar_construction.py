import json
import math
import logging
import argparse
from collections import Counter

from pyspark import SparkContext, SparkConf, StorageLevel

from hsst.utility import utility
from hsst.rulextraction.rule_filter import apply_filters
from hsst.rulextraction.rule_type import load_wmap, assign_rule_type


def read_file(sc, filename):
    logging.info('Loading dataset with filename {}'.format(filename))

    return sc.sequenceFile(filename).map(lambda x: utility.list_to_tuple(json.loads(x[1])))


def parse_rule(source_nodes, source_edges, target, dmrs_wmap, target_wmap):
    # Parse source and target sides and map to readable text
    source_nodes = [dmrs_wmap.value[node_id] if not node_id.startswith('X') else node_id for _, node_id in source_nodes]

    parsed_edges = []
    for edge in source_edges:
        parsed_edges.append([edge[0], edge[1], dmrs_wmap.value[edge[2]]])

    target = ' '.join([target_wmap.value[idx] if not idx.startswith('X') else idx for idx in target])

    return source_nodes, source_edges, target


def filter_rule(source, target, dmrs_wmap, target_wmap):
    source_nodes, source_edges = source
    source_nodes, source_edges, target = parse_rule(source_nodes, source_edges, target, dmrs_wmap, target_wmap)

    return apply_filters((source_nodes, source_edges, target)) == 'x_no_filter'


def compute_features(source, target, count, dmrs_wmap, target_wmap):
    source_nodes, source_edges = source

    feature_dict = {'count': count,
                    'node_count': len(source_nodes),
                    'edge_count': len(source_edges)}

    if dmrs_wmap is not None and target_wmap is not None:
        source_nodes, source_edges, target = parse_rule(source_nodes, source_edges, target, dmrs_wmap, target_wmap)
        feature_dict['rule_type'] = assign_rule_type(source_nodes, source_edges)

    return feature_dict


def compute_probabilities(counter, logprob=True):
    rule_probs = dict()
    key_count = sum(counter.values())

    for value in counter:
        if logprob:
            if key_count == 0 or counter[value] == 0:
                rule_probs[value] = 100.0
            else:
                # Adding zero to avoid negative zero
                rule_probs[value] = - math.log(counter[value] / float(key_count)) + 0.0
        else:
            if key_count == 0:
                rule_probs[value] = 0.0
            else:
                rule_probs[value] = counter[value] / float(key_count)

    return rule_probs


def process(ruleset_filename, grammar_filename, sc, num_partitions, dmrs_wmap=None, target_wmap=None, rule_filter=False):

    # Read rules from HDFS
    rules = read_file(sc, ruleset_filename).repartition(num_partitions)

    if rule_filter:
        rules = rules.filter(lambda x: filter_rule(x[0], x[1], dmrs_wmap, target_wmap))

    # Compute source to target probabilities
    s2t = rules.combineByKey(lambda x: Counter([x]), lambda c, x: c + Counter([x]), lambda c1, c2: c1 + c2).persist(StorageLevel.MEMORY_ONLY)
    s2t_prob = s2t.flatMap(lambda x: [((x[0], target), {'s2t_prob': prob}) for target, prob in compute_probabilities(x[1]).items()])

    # Compute target to source probabilities
    t2s = rules.map(lambda x: (x[1], x[0])).combineByKey(lambda x: Counter([x]), lambda c, x: c + Counter([x]), lambda c1, c2: c1 + c2)
    t2s_prob = t2s.flatMap(lambda x: [((source, x[0]), {'t2s_prob': prob}) for source, prob in compute_probabilities(x[1]).items()])

    # Join s2t and t2s probabilities into single feature dict
    rules_prob = s2t_prob.join(t2s_prob).map(lambda x: (x[0], dict(x[1][0].items() + x[1][1].items())))

    # Compute other features
    rules_feat = s2t.flatMap(lambda x: [((x[0], target), compute_features(x[0], target, count, dmrs_wmap, target_wmap)) for target, count in x[1].most_common()])

    # Combine the probability features with other features
    grammar_rules = rules_prob.join(rules_feat).map(lambda x: (x[0], dict(x[1][0].items() + x[1][1].items())))

    # Sort rules by number of nodes in the source side and then by alphabet, and add rule id as the index
    # ends up being (rule_id, (rule, feat_dict))
    grammar = grammar_rules.sortBy(lambda x: x[1]['count'], False).zipWithIndex().map(lambda x: utility.rule_to_string_grammar((x[1], x[0])))

    # Save grammar to disk
    grammar.saveAsTextFile(grammar_filename)


if __name__ == "__main__":

    appName = 'Grammar construction'

    parser = argparse.ArgumentParser(description='Do grammar construction with Spark.')
    parser.add_argument('-l', '--log')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-p', '--partitions', type=int, default=100)
    parser.add_argument('-t', '--rule_type', nargs=2, help='Specify DMRS and target language WMAP paths for rule type features')
    parser.add_argument('-f', '--rule_filter', action='store_true')
    parser.add_argument('output_filepath')
    parser.add_argument('input')

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

    grammar_out = 'file://' + args.output_filepath

    if args.rule_type:
        dmrs_wmap = sc.broadcast(load_wmap(args.rule_type[0]))
        target_wmap = sc.broadcast(load_wmap(args.rule_type[1]))
    else:
        dmrs_wmap = None
        target_wmap = None

    process(args.input, grammar_out, sc, num_partitions, dmrs_wmap, target_wmap, args.rule_filter)
