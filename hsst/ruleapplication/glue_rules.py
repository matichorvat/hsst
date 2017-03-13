from collections import defaultdict
from itertools import permutations
import xml.etree.ElementTree as xml

from hsst.utility.representation import loads
from hsst.utility.graph import read_graphs

DR = '999999999'


def string_create_sentence_glue_rules(sentence_xml_string, sentence_coverages, glue_rule_id_offset=100000000, idx=True, filtering=None, format='xml'):

    try:
        sentence_collection = loads(sentence_xml_string, format=format)
    except xml.ParseError:
        raise Exception('Error parsing sentence XML for id %d' % id)

    read_graphs(sentence_collection, idx=idx)

    sentence = sentence_collection[0]

    return create_sentence_glue_rules(sentence, sentence_coverages, glue_rule_id_offset, filtering)


def create_sentence_glue_rules(sentence, sentence_coverages, glue_rule_id_offset=100000000, filtering=None, max_deletion=1):

    if filtering is not None and not filtering.filter(sentence):
        return []

    if sentence.source.graph is None or sentence.source.dmrs is None:
        return []

    coverage_glue_rules = defaultdict(list)

    # Store each coverage for which a glue rule can be created according to # of nonterminals
    for coverage in sentence_coverages:
        glue_rule_type = compute_glue_rule_type(coverage, max_deletion=max_deletion)

        if glue_rule_type == 0:
            continue

        coverage_glue_rules[glue_rule_type].append(coverage)

    glue_rules = []

    # Create each glue rule type only once with multiple coverages
    for glue_rule_type, coverages in coverage_glue_rules.items():

        if glue_rule_type == 1:
            glue_rules.append((create_empty_glue_rule(glue_rule_id_offset), coverages))

        if glue_rule_type == 2:
            glue_rules.append((create_one_symbol_glue_rule(glue_rule_id_offset), coverages))

        if glue_rule_type == 3:
            glue_rules.append((create_two_symbol_glue_rule(glue_rule_id_offset), coverages))
            glue_rules.append((create_two_symbol_glue_rule(glue_rule_id_offset, reverse=True), coverages))

    glue_rule_3arg = create_3arg_glue_rules(sentence, sentence_coverages, glue_rule_id_offset)
    glue_rules.extend(glue_rule_3arg)

    return glue_rules


def compute_glue_rule_type(coverage, max_deletion=1):

    num_deleted_nodes = sum(1 for node in coverage if node == '1')

    if num_deleted_nodes > max_deletion:
        return 0

    num_nonterminals = len(set(node for node in coverage if node.startswith('X')))

    if num_nonterminals == 0:
        return 1

    elif num_nonterminals == 1:
        return 2

    elif num_nonterminals == 2:
        return 3

    return 0


def create_empty_glue_rule(glue_rule_id_offset):
    return str(glue_rule_id_offset), [], [DR], {'rule_type': 'glue', 'glue_type': '0'}


def create_one_symbol_glue_rule(glue_rule_id_offset):
    return str(glue_rule_id_offset + 1), [], [DR, 'X_0'], {'rule_type': 'glue', 'glue_type': '1'}


def create_two_symbol_glue_rule(glue_rule_id_offset, reverse=False):
    return (str(glue_rule_id_offset + 2) if not reverse else str(glue_rule_id_offset + 3),
            [],
            [DR, 'X_0', 'X_1'] if not reverse else [DR, 'X_1', 'X_0'],
            {'rule_type': 'glue', 'glue_type': '2'})


def create_3arg_glue_rules(sentence, sentence_coverages, glue_rule_id_offset):
    reference_coverage = sorted([node.node_id for node in sentence.source.graph.nodes])
    node_3args = find_3arg_nodes(sentence)

    full_glue_rules = []

    coverages = []

    for node_id, target_node_ids in node_3args:
        coverage = create_3arg_coverage(node_id, target_node_ids, reference_coverage, sentence_coverages)

        if coverage is not None:
            coverages.append(coverage)

    if coverages:
        glue_rules = create_all_three_symbol_glue_rules(glue_rule_id_offset)
        for glue_rule in glue_rules:
            full_glue_rules.append((glue_rule, coverages))

    return full_glue_rules


def find_3arg_nodes(sentence):
    """
    Find nodes in sentence DMRS that have 3 outgoing links
    :param sentence: Sentence object
    :return: List of tuples containing 3arg node id and a list of target nodes
    """
    out_dict = defaultdict(set)
    for element in sentence.source.dmrs:
        if element.tag == 'link':
            out_dict[element.attrib['from']].add(element.attrib['to'])

    return [(node_id, to_node_ids) for node_id, to_node_ids in out_dict.items() if len(to_node_ids) == 3]


def create_3arg_coverage(node_id, target_node_ids, reference_coverage, sentence_coverages):
    coverage = ['0'] * len(reference_coverage)
    coverage[reference_coverage.index(node_id)] = '1'

    nonterminal_index = 0

    for target_node_id in target_node_ids:
        target_node_id_index = reference_coverage.index(target_node_id)

        highest_subpart_coverage = None
        highest_subpart_coverage_len = 0

        for existing_coverage in sentence_coverages:
            coverage_len = sum(1 for x in existing_coverage if x != '0')

            if existing_coverage[target_node_id_index] == '0':
                continue

            if existing_coverage[reference_coverage.index(node_id)] != '0':
                continue

            if coverage_len > highest_subpart_coverage_len:
                highest_subpart_coverage_len = coverage_len
                highest_subpart_coverage = existing_coverage

        if highest_subpart_coverage is None:
            return None

        update_3arg_coverage(coverage, highest_subpart_coverage, nonterminal_index)
        nonterminal_index += 1

    assert 'X_0' in coverage and 'X_1' in coverage and 'X_2' in coverage

    return tuple(coverage)


def update_3arg_coverage(coverage, existing_coverage, nonterminal_index):
    for index, elem in enumerate(existing_coverage):
        if elem == '1' or elem.startswith('X'):
            coverage[index] = 'X_%s' % (nonterminal_index,)


def create_all_three_symbol_glue_rules(glue_rule_id_offset):
    glue_rules = []

    id_offset = glue_rule_id_offset + 4

    for target_side in permutations(['X_0', 'X_1', 'X_2']):
        glue_rules.append((str(id_offset), [], [DR] + list(target_side), {'rule_type': 'glue', 'glue_type': '3'}))
        id_offset += 1

    return glue_rules
