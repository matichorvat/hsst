import xml.etree.ElementTree as xml
from itertools import permutations, chain

from hsst.utility.representation import loads
from hsst.utility.graph import read_graphs


def string_create_sentence_mapping_rules(sentence_xml_string, sentence_coverages, mapping_rule_id_offset=130000000, idx=True, filtering=None, format='xml'):

    try:
        sentence_collection = loads(sentence_xml_string, format=format)
    except xml.ParseError:
        raise Exception('Error parsing sentence XML for id %d' % id)

    read_graphs(sentence_collection, idx=idx)

    sentence = sentence_collection[0]

    return create_sentence_mapping_rules(sentence, sentence_coverages, mapping_rule_id_offset, filtering)


def create_sentence_mapping_rules(sentence, sentence_coverages, mapping_rule_id_offset, filtering=None):

    if filtering is not None and not filtering.filter(sentence):
        return sentence.id, []

    if sentence.source.graph is None or sentence.source.dmrs is None:
        return sentence.id, []

    reference_coverage = sorted([node.node_id for node in sentence.source.graph.nodes])

    mapping_nodes = get_mapping_nodes(sentence)

    mapping_rules = []

    index = 0
    for mapping_node in mapping_nodes:
        mapping_node_list, target_side = mapping_node

        mapping_coverages = find_mapping_coverages(mapping_node_list, reference_coverage, sentence_coverages)

        for mapping_coverage in mapping_coverages:
            # Create a globally unique rule id, based on the current sentence id
            rule_id = mapping_rule_id_offset + int(sentence.id) * 100 + index

            rules = create_mapping_rules(rule_id, target_side, mapping_coverage)
            mapping_rules.extend(rules)

            index += len(rules)

    return mapping_rules


def get_mapping_nodes(sentence):
    dmrs = sentence.source.dmrs

    mapping_nodes = []

    for element in dmrs:
        if not element.tag == 'node' or element.attrib.get('carg_idx') or element.attrib.get('label') == 'compound':
            continue

        node_mapping = element.attrib.get('tok_idx')

        if node_mapping is None or node_mapping == '':
            continue

        target_side = node_mapping.split(' ')

        assert len(target_side) >= 1

        mapping_nodes.append(([element.attrib.get('nodeid')], target_side))

    return mapping_nodes


def create_mapping_rules(rule_id, mapping_value, mapping_coverage):
    rules = []

    num_nonterminals = len(set(node for node in mapping_coverage if str(node).startswith('X')))

    if num_nonterminals == 0:
        rule = str(rule_id), [], mapping_value, {'rule_type': 'mapping_terminal'}
        rules.append((rule, [mapping_coverage]))
        return rules

    # If there are nonterminals, create all permutations of target side
    for target_side_permutation in permutations(range(0, num_nonterminals + 1)):
        # Replace carg_value index with multiple elements of carg_values (e.g. (0,1,2)->(X_0,X_1,123,42)
        target_side = list(
            chain.from_iterable(
                tuple(mapping_value) if index >= num_nonterminals else ('X_%d' % index,) for index in
                target_side_permutation
            )
        )

        rule = str(rule_id + len(rules)), [], target_side, {'rule_type': 'mapping_nonterminal'}
        rules.append((rule, [mapping_coverage]))

    return rules


def find_mapping_coverages(mapping_node_list, reference_coverage, coverages):
    """
    Find mapping coverages by searching coverages for the ones with 1 in exactly the places mapping requires it to be and nowhere else.
    e.g. mapping is in node_id 2, reference coverage is [0,1,2,3], mapping coverages are [0,0,1,X0], [0,0,1,0] but not [0,1,1,0] or [0,1,X0,0]
    :param mapping_node_list: List of node ids corresponding to a single mapping
    :param reference_coverage: List of node ids in order which coverages are constructed
    :param coverages: List of coverages obtained by running rule application
    :return: List of coverages corresponding to the give mapping
    """

    # mapping node positions in reference
    mapping_node_positions = [reference_coverage.index(node_id) for node_id in mapping_node_list]

    mapping_coverages = []

    # Search rule application coverages that have 1 in exactly those coverages and nowhere else
    for coverage in coverages:
        if not all([True if coverage[index] == '1' else False for index in mapping_node_positions]):
            continue

        if sum(1 for node in coverage if node == '1') != len(mapping_node_positions):
            continue

        mapping_coverages.append(coverage)

    return mapping_coverages
