import os
import re
import logging
import argparse
from collections import defaultdict

from pyspark import SparkContext, SparkConf, StorageLevel

from hsst.utility import utility
from hsst.ruleapplication.carg_rules import string_create_sentence_carg_rules
from hsst.ruleapplication.glue_rules import string_create_sentence_glue_rules
from hsst.ruleapplication.disc_rules import string_create_sentence_disc_rules
from hsst.ruleapplication.mapping_rules import string_create_sentence_mapping_rules
from hsst.preprocessing.filtering import SourceTokLength, SourceGraphSizeFilter
from hsst.ruleapplication.ruleapplication_interface import string_sentence_ruleapplication


def read_file(sc, filename, num_partitions):
    # Split with custom delimiter
    dataset = sc.newAPIHadoopFile(
        filename,
        'org.apache.hadoop.mapreduce.lib.input.TextInputFormat',
        'org.apache.hadoop.io.Text',
        'org.apache.hadoop.io.LongWritable',
        conf={'textinputformat.record.delimiter': '</sentence>'}
    )

    # Reattach the delimiter to each record
    return dataset.filter(lambda x: x[1].strip() != '').map(lambda x: x[1] + '</sentence>').repartition(num_partitions)


def source_side_to_text(source_side):
    src_nodes = [(str(x.node_id), x.label) for x in sorted(source_side.nodes, key=lambda x: x.node_id)]
    src_edges = [(str(x.from_node.node_id), str(x.to_node.node_id), x.label) for x in sorted(source_side.edges, key=lambda x: x.edge_id)]
    return tuple(src_nodes), tuple(src_edges)


def rule_application(data, filtering=None, timeout=None):
    rule_app = data.flatMap(lambda x: rule_application_sentence(x, filtering, timeout))
    return rule_app


def get_id_from_xml(sentence_xml):
    m = re.search('sentence id="([0-9]+)" orig', sentence_xml)

    if m:
        return m.group(1)
    else:
        return -1


def create_glue_rules(sentence_xml_and_coverages, idx=True, filtering=None):
    # Map to (sent_id, [(glue_rule, coverages), ...])
    sentence_glue_rules = sentence_xml_and_coverages.map(lambda x: string_create_sentence_glue_rules(x[1][0], x[1][1], idx=idx, filtering=filtering))

    return sentence_glue_rules


def create_carg_rules(sentence_xml_and_coverages, idx=True, filtering=None):
    # Map to (sent_id, [(carg_rule, coverages), ...])
    sentence_carg_rules = sentence_xml_and_coverages.map(lambda x: string_create_sentence_carg_rules(x[1][0], x[1][1], idx=idx, filtering=filtering))
    return sentence_carg_rules


def create_disc_rules(sentence_xml_and_coverages, idx=True, filtering=None):
    # Map to (sent_id, [(disc_rule, coverages), ...])
    sentence_disc_rules = sentence_xml_and_coverages.map(lambda x: string_create_sentence_disc_rules(x[1][0], x[1][1], idx=idx, filtering=filtering))

    return sentence_disc_rules


def create_mapping_rules(sentence_xml_and_coverages, idx=True, filtering=None):
    # Map to (sent_id, [(mapping_rule, coverages), ...])
    sentence_mapping_rules = sentence_xml_and_coverages.map(lambda x: string_create_sentence_mapping_rules(x[1][0], x[1][1], idx=idx, filtering=filtering))
    return sentence_mapping_rules


def rule_application_sentence(sentence_xml_string, filtering=None, timeout=None):
    results = list()

    logging.info('Starting processing of sentence: {}'.format(sentence_xml_string[:50]))

    sent_id, applied_rules = string_sentence_ruleapplication(
        sentence_xml_string,
        filtering=filtering,
        timeout=timeout
    )

    for source_side, coverages in applied_rules.items():
        results.append((source_side_to_text(source_side), (sent_id, coverages)))

    return results


def save_partition(
        elements,
        reference_grammar,
        sentence_grammar_path,
        glue_rule_opt=False,
        carg_rule_opt=False,
        disc_rule_opt=False,
        mapping_rule_opt=False):

    reference_grammar = reference_grammar.value

    for element in elements:
        # Element = (sent_id, (grammar_rule_list, non_grammar_rule_list))
        save_sentence_rules(
            element[0],
            element[1][0],
            element[1][1],
            reference_grammar,
            sentence_grammar_path,
            glue_rule_opt=glue_rule_opt,
            carg_rule_opt=carg_rule_opt,
            disc_rule_opt=disc_rule_opt,
            mapping_rule_opt=mapping_rule_opt
        )


def save_sentence_rules(
        sentence_id,
        rule_and_coverages_list,
        sentence_xml_and_coverages,
        reference_grammar,
        output_path,
        glue_rule_opt=False,
        carg_rule_opt=False,
        disc_rule_opt=False,
        mapping_rule_opt=False):

    logging.info(output_path)

    coverage_map = defaultdict(list)

    sentence_xml, all_coverages = sentence_xml_and_coverages

    non_grammar_rules = []
    if glue_rule_opt:
        non_grammar_rules.extend(string_create_sentence_glue_rules(sentence_xml, all_coverages))

    if carg_rule_opt:
        non_grammar_rules.extend(string_create_sentence_carg_rules(sentence_xml, all_coverages))

    if disc_rule_opt:
        non_grammar_rules.extend(string_create_sentence_disc_rules(sentence_xml, all_coverages))

    if mapping_rule_opt:
        non_grammar_rules.extend(string_create_sentence_mapping_rules(sentence_xml, all_coverages))

    with open(os.path.join(output_path, '%s.grammar' % sentence_id), 'wb') as f:
        logging.info(os.path.join(output_path, '%s.grammar' % sentence_id))
        logging.debug(rule_and_coverages_list)

        if rule_and_coverages_list is not None:
            for rule_id, coverages in sorted(rule_and_coverages_list, key=lambda x: int(x[0])):
                rule_string = reference_grammar[rule_id]

                rule_string = utility.rule_to_string_rule_app(utility.grammar_rule_string_to_tuple(rule_string))
                f.write('%s\n' % (rule_string,))

                for coverage in coverages:
                    coverage_string = coverage_to_string(coverage)
                    coverage_map[coverage_string].append(rule_id)

        for rule, coverages in sorted(non_grammar_rules, key=lambda x: int(x[0][0])):
            rule_string = utility.rule_to_string_rule_app(rule)

            f.write('%s\n' % (rule_string,))
            rule_id, _, _, _ = rule

            for coverage in coverages:
                coverage_string = coverage_to_string(coverage)
                coverage_map[coverage_string].append(rule_id)

    with open(os.path.join(output_path, '%s.coverage' % sentence_id), 'wb') as f:
        logging.info(os.path.join(output_path, '%s.coverage' % sentence_id))

        for coverage, rule_ids in sorted(coverage_map.items(), key=lambda x: coverage_sort(x[0])):
            f.write('%s\t%s\n' % (coverage, ','.join(str(x) for x in rule_ids)))


def coverage_sort(coverage):
    return sum(1 for y in coverage.split('_') if y != '0'), -sum(1 for y in coverage.split('_') if y == '1')


def coverage_to_string(coverage):

    single_coverage_list = list()

    for bit in coverage:
        if bit.startswith('X'):
            bit = bit.replace('_', '')

        single_coverage_list.append(bit)

    return '_'.join(single_coverage_list)


def process(
        test_set_filename,
        grammar_filename,
        sentence_grammar_path,
        sc,
        num_partitions,
        filtering=None,
        timeout=None,
        glue_rule_opt=False,
        carg_rule_opt=False,
        disc_rule_opt=False,
        mapping_rule_opt=False):

    # Create reference grammar
    reference_grammar = utility.load_hsst_grammar(grammar_filename if not grammar_filename.startswith('file:') else grammar_filename[5:])

    # Broadcast reference grammar so it's accessible from every executioner
    reference_grammar = sc.broadcast(reference_grammar)

    # Load intersection grammar
    # --> (rule_id, source_side, target_side, feature_dict)
    intersection_grammar = sc.textFile(grammar_filename, minPartitions=num_partitions).map(utility.grammar_rule_string_to_tuple)

    # --> (source_side, rule_id)
    intersection_grammar = intersection_grammar.map(lambda rule: (rule[1], rule[0]))

    # Load dataset
    # --> (sentence_xml_string)
    data = read_file(sc, test_set_filename, num_partitions).persist(StorageLevel.MEMORY_ONLY)  #.sample(False, 0.1, 1)

    # Perform rule application
    # --> (source_side, (sent_id, coverages)))
    rule_app = rule_application(data, filtering=filtering, timeout=timeout).persist(StorageLevel.MEMORY_AND_DISK)

    # Intersect grammar with rule application source sides
    # --> (source_side, ((sent_id, coverages), rule_id))
    sentence_rules = rule_app.join(intersection_grammar)

    # Map resulting sentence rules in appropriate key:value
    # --> (sent_id, (rule_id, coverages))
    sentence_rules = sentence_rules.map(lambda x: (x[1][0][0], (x[1][1], x[1][0][1])))

    # Group resulting sentence rules by sentence id
    # --> (sent_id, [(rule_id, coverages), ...])
    sentence_rules = sentence_rules.groupByKey()

    # Create structures for non-grammar rules

    # Map data
    # --> (sent_id, sentence_xml_string)
    sentence_xmls = data.map(lambda x: (get_id_from_xml(x), x))

    # Merge all sent_id coverages from rule application into a single list
    # --> (sent_id, all_coverages_list)
    sentence_coverages = rule_app.map(lambda x: (x[1][0], x[1][1])).reduceByKey(
        lambda a, b: list(a) + list(b) if a and b else list(b) if list(b) else list(a))

    # Join (sent_id, sentence_xml_string) and (sent_id, all_coverages_list)
    # --> (sent_id, (sentence_xml, all_coverages_list))
    sentence_xml_and_coverages = sentence_xmls.join(sentence_coverages)

    # Join (sent_id, (sentence_xml, all_coverages_list)) to sentence_rules
    # --> (sent_id, ([(rule_id, coverages), ...], (sentence_xml, all_coverages_list)))
    sentence_rules = sentence_rules.fullOuterJoin(sentence_xml_and_coverages)

    # Save each sentence grammar in a separate file
    sentence_rules.repartition(5).foreachPartition(lambda elements: save_partition(
        elements,
        reference_grammar,
        sentence_grammar_path,
        glue_rule_opt=glue_rule_opt,
        carg_rule_opt=carg_rule_opt,
        disc_rule_opt=disc_rule_opt,
        mapping_rule_opt=mapping_rule_opt
    ))


if __name__ == "__main__":

    # Sample paths:
    # grammar_in = 'file:///data/mifs_scratch/mh693/wmt15/0132-wmt15-en-de/rule_extract/hsst.grammar.idx.nofilter.txt'
    # set_grammar_out = 'file:///data/mifs_scratch/mh693/wmt15/0132-wmt15-en-de/rule_app/grammar.set'
    #
    # test_in = '/data/mifs_scratch/mh693/wmt15/0132-wmt15-en-de/rule_app/test.ds'
    # rule_app_out = 'file:///data/mifs_scratch/mh693/wmt15/0132-wmt15-en-de/rule_app/rule.app'

    appName = 'Rule Application'

    parser = argparse.ArgumentParser(description='Do ruleapplication with Spark.')
    parser.add_argument('-l', '--log')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-p', '--partitions', type=int, default=100)
    parser.add_argument('-m', '--mode', default='sgtt', choices=['sgtt', 'sttg', 'sgtg', 'sttt'])
    parser.add_argument('-t', '--timeout', type=int, default=0)
    parser.add_argument('-f', '--filter', choices=['token_num', 'graph_size'])
    parser.add_argument('-fmax', '--filtermax', type=int, default=20)
    parser.add_argument('-fmin', '--filtermin', type=int, default=0)
    parser.add_argument('-g', '--glue_rules', action='store_true')
    parser.add_argument('-c', '--carg_rules', action='store_true')
    parser.add_argument('-d', '--disc_rules', action='store_true')
    parser.add_argument('--mapping_rules', action='store_true')
    parser.add_argument('dataset_filepath')
    parser.add_argument('grammar_filepath')
    parser.add_argument('output_filepath')

    args = parser.parse_args()

    # Set up logging
    if args.log:
        logging.basicConfig(filename=args.log, level=logging.DEBUG if args.verbose else logging.INFO)
    else:
        logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    num_partitions = args.partitions

    conf = SparkConf().setAppName(appName).set('spark.executor.extraJavaOptions', '-Dfile.encoding=UTF-8')
    sc = SparkContext(conf=conf)

    if args.mode == 'sgtt':
        if args.filter == 'token_num':
            filtering = SourceTokLength(max_len=args.filtermax, min_len=args.filtermin)
        elif args.filter == 'graph_size':
            filtering = SourceGraphSizeFilter(max_nodes=args.filtermax, min_nodes=args.filtermin)
        else:
            filtering = None

    else:
        logging.exception('Modes other than sgtt not implemented.')
        raise NotImplementedError('Modes other than sgtt not implemented.')

    dataset = 'file:' + args.dataset_filepath
    grammar = 'file:' + args.grammar_filepath
    sentence_grammar_path = args.output_filepath

    process(
        dataset,
        grammar,
        sentence_grammar_path,
        sc,
        num_partitions,
        filtering=filtering,
        timeout=args.timeout,
        glue_rule_opt=args.glue_rules,
        carg_rule_opt=args.carg_rules,
        disc_rule_opt=args.disc_rules,
        mapping_rule_opt=args.mapping_rules
    )
