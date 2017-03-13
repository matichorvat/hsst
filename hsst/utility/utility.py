import os
import ast
import gzip
import errno
import shutil
import random
import logging
import resource
import itertools
from subprocess import check_call, Popen

from hsst.utility import representation


def open_file(filename):
    if filename is not None:
        if filename.endswith('.gz'):
            return gzip.open(filename, 'rb')
        else:
            return open(filename, 'rb')
    else:
        return None


def close_file(file_handle):
    if file_handle is not None:
        file_handle.close()


def create_alignment_dict(alignments, graph):
    alignment_dict = dict()

    # Connect node ids in alignment to node objects in the graph
    for node_id in alignments:
        alignment = alignments[node_id]

        for node in graph.nodes:
            if node.node_id == node_id:
                alignment_dict[node] = alignment
                break

    return alignment_dict


def create_realization_alignment(dmrs):
    alignments = dict()

    for element in dmrs:

        if element.tag == 'node':
            node_id = element.attrib.get('nodeid')
            tokalign = element.attrib.get('tokalign')

            if node_id is None or tokalign is None:
                continue

            alignment = list()
            for token in [x for x in tokalign.split(' ') if x is not None and x.strip() != '']:
                try:
                    alignment.append(int(token))
                except ValueError as e:
                    logging.warn('ValueError for token %s (%s)' % (token, e))
                    continue

            alignments[node_id] = alignment

    return alignments


def list_to_tuple(x):
    if isinstance(x, list):
        return tuple(list_to_tuple(y) for y in x)
    else:
        return x


def pairwise(iterable):
    """
    s -> (s0,s1), (s1,s2), (s2, s3), ...
    """
    a, b = itertools.tee(iterable)
    next(b, None)
    return itertools.izip(a, b)


def get_carg_rules(dmrs):

    all_nodes = list()
    carg_nodes = dict()

    for element in dmrs:

        if element.tag == 'node':
            node_id = element.attrib.get('nodeid')
            all_nodes.append(node_id)

            node_carg = element.attrib.get('carg_idx')

            if node_carg is not None:
                carg_nodes[node_id] = list_to_tuple(unicode(node_carg).split(' '))

    reference_coverage = sorted(all_nodes)

    carg_rules = list()

    for node_id in carg_nodes:
        coverage = [0] * len(reference_coverage)
        coverage[reference_coverage.index(node_id)] = 1
        coverage = tuple(coverage)

        carg_rules.append((coverage, carg_nodes[node_id]))

    return carg_rules


def int_convert(coverage):
    return [int(x) if x == '0' or x == '1' else x for x in coverage]


def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            logging.exception("Error creating directory: %s" % path)
            raise


def remove_directory(directory):
    try:
        shutil.rmtree(directory)
    except OSError:
        pass


def copy_directory(src, dst):
    shutil.copytree(src, dst)


def create_dir(directory):
    #remove_directory(directory)
    make_sure_path_exists(directory)


def move_directory(src, dst, gzip=False):
    #remove_directory(dst)

    if gzip:
        for filename in os.listdir(src):
            src_filepath = os.path.join(src, filename)
            check_call(['gzip', '-f', src_filepath])

    copytree(src, dst)
    remove_directory(src)


def move_and_rename_files(src, dst, mapping, gzip=False):
    assert src != dst

    make_sure_path_exists(dst)

    for filename in os.listdir(src):
        src_filepath = os.path.join(src, filename)
        trg_filepath = os.path.join(dst, mapping[filename])
        shutil.move(src_filepath, trg_filepath)

        if gzip:
            check_call(['gzip', '-f', trg_filepath])


def gzip_file(filename, overwrite=True):
    if overwrite:
        check_call(['gzip', '-f', filename])
    else:
        check_call(['gzip', filename])


def ungzip_file(filename, output):
    cmd = 'gzip -d -c %s > %s' % (filename, output)
    logging.debug(cmd)
    p = Popen(cmd, shell=True)
    p.wait()


def gzip_directory(directory, overwrite=True):
    if overwrite:
        check_call(['gzip', '-f', '-r', directory])
    else:
        check_call(['gzip', '-r', directory])


def copytree(src, dst, symlinks=False, ignore=None):
    if not os.path.exists(dst):
        os.makedirs(dst)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            copytree(s, d, symlinks, ignore)
        else:
            if not os.path.exists(d) or os.stat(s).st_mtime - os.stat(d).st_mtime > 1:
                shutil.copy2(s, d)


def merge_datasets(directory, output_filename, encode=False, filtering=None, max_num=None):

    dataset_filenames = [os.path.join(directory, filename) for filename in os.listdir(directory)]

    merged_dataset = representation.SentenceCollection()
    new_id = 0

    for dataset_filename in sorted(dataset_filenames):

        ds = open_file(dataset_filename)
        dataset = representation.load(ds)
        close_file(ds)

        for sentence in dataset:
            if filtering is not None and not filtering.filter(sentence):
                continue

            sentence.id = new_id
            merged_dataset.add(sentence)
            new_id += 1

    if max_num is not None:
        lim_dataset = representation.SentenceCollection()
        x = merged_dataset.sentences
        random.shuffle(x)
        new_id = 0
        while new_id < max_num:
            sentence = x[new_id]
            sentence.id = new_id
            lim_dataset.add(sentence)
            new_id += 1

        merged_dataset = lim_dataset

    with open(output_filename, 'wb') as mds:
        representation.dump(merged_dataset, mds, encode=encode)

    gzip_file(output_filename)


def load_dataset(filename, subset_range=None):
    fp = open_file(filename)
    dataset = representation.load(fp, format='xml', subset_range=subset_range)
    close_file(fp)

    return dataset


def load_hsst_grammar(filename):
    logging.info('Loading HSST grammar.')

    grammar = dict()

    gf = open_file(filename)

    for line in gf:
        line = line.strip()
        if line == '':
            continue

        grammar[line.split('\t')[0]] = line

    close_file(gf)

    logging.info('HSST grammar with %d rules loaded.' % (len(grammar),))

    return grammar


def rule_to_string_rule_app(rule):
    rule_id, source_side, target_side, feature_dict = rule

    target_string = create_target_string(target_side)

    return '%s\t%s\t%s' % (rule_id, target_string, feature_dict)


def rule_to_string_grammar(rule):
    rule_id, rule_parts = rule
    source_and_target, feature_dict = rule_parts
    source_side, target_side = source_and_target

    source_string = create_source_string(source_side)
    target_string = create_target_string(target_side)

    source_and_target_string = '%s\t%s' % (source_string, target_string)

    return '%s\t%s\t%s' % (rule_id, source_and_target_string, feature_dict)


def create_source_string(source_tuple):
    nodes, edges = source_tuple

    node_list = list()
    edge_list = list()

    count = 0
    for node_id, node in sorted(nodes):
        assert int(node_id) == count

        if node.startswith('X'):
            node = node.replace('_', '')

        node_list.append(node)
        count += 1

    for edge in edges:
        edge_list.append('%s-%s-%s' % edge)

    node_string = '_'.join(node_list)
    edge_string = '_'.join(edge_list)

    return '%s|%s' % (node_string, edge_string)


def create_target_string(target_tuple):
    target_list = list()

    for token in target_tuple:
        if token.startswith('X'):
            token = token.replace('_', '')

        target_list.append(token)

    return '_'.join(target_list)


def rule_string_to_tuple(rule_id, target_string, feature_dict, source_string=None):
    target_side = tuple(x if not x.startswith('X') else x.replace('X', 'X_') for x in target_string.split('_'))

    if source_string is not None:
        source_node_string, source_edge_string = source_string.split('|')
        source_side_nodes = tuple((str(index), x) if not x.startswith('X') else (str(index), x.replace('X', 'X_')) for index, x in enumerate(source_node_string.split('_')))

        if source_edge_string != '':
            source_side_edges = tuple(tuple(edge.split('-')) for edge in source_edge_string.split('_'))
        else:
            source_side_edges = ()

        rule = (rule_id, (source_side_nodes, source_side_edges), target_side, ast.literal_eval(feature_dict))

    else:
        rule = (rule_id, target_side, ast.literal_eval(feature_dict))

    return rule


def grammar_rule_string_to_tuple(rule_string):
    rule_parts = rule_string.split('\t')

    assert len(rule_parts) == 4

    rule_id, source_string, target_string, feature_dict = rule_parts
    rule = rule_string_to_tuple(rule_id, target_string, feature_dict, source_string=source_string)

    return rule


def load_sentence_specific_grammar(sentence_id, sentence_grammar_pattern):

    sentence_grammar = dict()
    filepath = sentence_grammar_pattern.replace('?', str(sentence_id))

    if not os.path.isfile(filepath):
        logging.warn('Sentence specific grammar for sentence %s does not exist at %s.' % (sentence_id, filepath))
        return sentence_grammar

    with open(filepath, 'rb') as f:
        for line in f:
            line = line.strip()

            if len(line) == 0:
                continue

            line_parts = line.split('\t')

            assert len(line_parts) == 3

            rule_id, target_string, feature_dict = line_parts
            rule = rule_string_to_tuple(int(rule_id), target_string, feature_dict)

            sentence_grammar[int(rule_id)] = rule

    return sentence_grammar


def load_sentence_specific_coverage(sentence_id, sentence_coverage_pattern):

    coverages = dict()
    filepath = sentence_coverage_pattern.replace('?', str(sentence_id))

    if not os.path.isfile(filepath):
        logging.warn('Sentence specific coverage for sentence %s does not exist at %s.' % (sentence_id, filepath))
        return coverages

    with open(filepath, 'rb') as f:
        for line in f:
            line = line.strip()

            if len(line) == 0:
                continue

            line_parts = line.split('\t')

            assert len(line_parts) == 2

            coverage_string, rule_ids_string = line_parts

            coverage = tuple(int(x) if not x.startswith('X') else x.replace('X', 'X_') for x in coverage_string.split('_'))
            rule_ids_list = [int(x) for x in rule_ids_string.split(',')]

            coverages[coverage] = rule_ids_list

    return coverages


ONE_GB = 1024 * 1024 * 1024


def set_memory_limit(memory_limit):
    rsrc = resource.RLIMIT_AS
    soft, hard = resource.getrlimit(rsrc)

    if memory_limit == -1:
        value = -1
    else:
        value = memory_limit * ONE_GB

    # Unit is in KBs
    resource.setrlimit(rsrc, (value, hard))
    soft, hard = resource.getrlimit(rsrc)
    logging.debug('Memory limit set to %d GB: %d' % (memory_limit, soft))