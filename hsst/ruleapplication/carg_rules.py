import xml.etree.ElementTree as xml
from itertools import permutations, chain

from hsst.utility.representation import loads
from hsst.utility.graph import read_graphs


def string_create_sentence_carg_rules(sentence_xml_string, sentence_coverages, carg_rule_id_offset=110000000, idx=True, filtering=None, format='xml'):

    try:
        sentence_collection = loads(sentence_xml_string, format=format)
    except xml.ParseError:
        raise Exception('Error parsing sentence XML for id %d' % id)

    read_graphs(sentence_collection, idx=idx)

    sentence = sentence_collection[0]

    return create_sentence_carg_rules(sentence, sentence_coverages, carg_rule_id_offset, filtering)


def create_sentence_carg_rules(sentence, sentence_coverages, carg_rule_id_offset, filtering=None):

    if filtering is not None and not filtering.filter(sentence):
        return []

    if sentence.source.graph is None or sentence.source.dmrs is None:
        return []

    reference_coverage = sorted([node.node_id for node in sentence.source.graph.nodes])

    carg_nodes = get_carg_nodes(sentence)

    carg_rules = []

    index = 0
    for carg_node in carg_nodes:
        carg_node_list, target_side = carg_node

        carg_coverages = find_carg_coverages(carg_node_list, reference_coverage, sentence_coverages)

        for carg_coverage in carg_coverages:
            # Create a globally unique rule id, based on the current sentence id
            rule_id = carg_rule_id_offset + int(sentence.id) * 100 + index

            rules = create_carg_rules(rule_id, target_side, carg_coverage)
            carg_rules.extend(rules)

            index += len(rules)

    return carg_rules


def get_carg_nodes(sentence):
    dmrs = sentence.source.dmrs

    carg_nodes = []

    for element in dmrs:
        if not element.tag == 'node':
            continue

        node_carg = element.attrib.get('carg_idx')

        if node_carg is None:
            continue

        target_side = node_carg.split(' ')

        assert len(target_side) >= 1

        carg_nodes.append(([element.attrib.get('nodeid')], target_side))

    return carg_nodes


def create_carg_rules(rule_id, carg_value, carg_coverage):

    rules = []

    num_nonterminals = len(set(node for node in carg_coverage if str(node).startswith('X')))

    if num_nonterminals == 0:
        rule = str(rule_id), [], carg_value, {'rule_type': 'carg_terminal'}
        rules.append((rule, [carg_coverage]))
        return rules

    # If there are nonterminals, create all permutations of target side
    for target_side_permutation in permutations(range(0, num_nonterminals + 1)):
        # Replace carg_value index with multiple elements of carg_values (e.g. (0,1,2)->(X_0,X_1,123,42)
        target_side = list(
            chain.from_iterable(
                tuple(carg_value) if index >= num_nonterminals else ('X_%d' % index,) for index in target_side_permutation
            )
        )

        rule = str(rule_id + len(rules)), [], target_side, {'rule_type': 'carg_nonterminal'}
        rules.append((rule, [carg_coverage]))

    return rules


def find_carg_coverages(carg_node_list, reference_coverage, coverages):
    """
    Find CARG coverages by searching coverages for the ones with 1 in exactly the places carg requires it to be and nowhere else.
    e.g. CARG is in node_id 2, reference coverage is [0,1,2,3], CARG coverages are [0,0,1,X0], [0,0,1,0] but not [0,1,1,0] or [0,1,X0,0]
    :param carg_node_list: List of node ids corresponding to a single CARG
    :param reference_coverage: List of node ids in order which coverages are constructed
    :param coverages: List of coverages obtained by running rule application
    :return: List of coverages corresponding to the give CARG
    """

    # CARG node positions in reference
    carg_node_positions = [reference_coverage.index(node_id) for node_id in carg_node_list]

    carg_coverages = []

    # Search rule application coverages that have 1 in exactly those coverages and nowhere else
    for coverage in coverages:
        if not all([True if coverage[index] == '1' else False for index in carg_node_positions]):
            continue

        if sum(1 for node in coverage if node == '1') != len(carg_node_positions):
            continue

        carg_coverages.append(coverage)

    return carg_coverages
