from itertools import permutations
import xml.etree.ElementTree as xml

from hsst.utility.representation import loads
from hsst.utility.graph import read_graphs


def string_create_sentence_disc_rules(sentence_xml_string, _, disc_rule_id_offset=120000000, idx=True, filtering=None, format='xml'):

    try:
        sentence_collection = loads(sentence_xml_string, format=format)
    except xml.ParseError:
        raise Exception('Error parsing sentence XML for id %d' % id)

    read_graphs(sentence_collection, idx=idx)

    sentence = sentence_collection[0]

    return create_sentence_disc_rules(sentence, disc_rule_id_offset, filtering)


def create_sentence_disc_rules(sentence, disc_rule_id_offset=120000000, filtering=None):

    if filtering is not None and not filtering.filter(sentence):
        return []

    if sentence.source.graph is None or sentence.source.dmrs is None:
        return []

    # Find disconnected graph parts and if the graph is not disconnected, return empty list
    disconnected_graph_parts = sentence.source.graph.find_disconnected_subgraphs()

    if not disconnected_graph_parts or len(disconnected_graph_parts) > 3:
        return []

    # assert len(disconnected_graph_parts) <= 2  # Protecting against combinatorial explosion

    disc_rules = []

    # Compute combined coverage (consisting only of non-terminals)
    disconnected_coverage = compute_disc_coverage(sentence, disconnected_graph_parts)
    coverages = [disconnected_coverage]

    feature_dict = {'rule_type': 'disc'}

    # Permute all possible target non-terminal sequences
    nonterminal_index_sequences = permutations(range(0, len(disconnected_graph_parts) + 1))

    # Create a disc rule for each target non-terminal sequence
    for index, nonterminal_index_sequence in enumerate(nonterminal_index_sequences):
        target_side = ['X_%d' % (nonterminal_index,) for nonterminal_index in nonterminal_index_sequence]

        rule = (str(disc_rule_id_offset + index), [], target_side, feature_dict)
        disc_rules.append((rule, coverages))

    return disc_rules


def compute_disc_coverage(sentence, disconnected_graph_parts):
    """
    Construct a disconnected glue rule coverage, e.g. X0_X0_X0_X0_X1_X1, where X0 is main graph part coverage
    :param sentence: Sentence object
    :param disconnected_graph_parts: List of disconnected graph parts, formed as sets of Node objects
    :return: Tuple coverage
    """

    part_coverages = compute_part_coverages(sentence, disconnected_graph_parts)

    disconnected_coverage = ['0'] * len(part_coverages[0])

    nonterminal_index = 0

    for part_coverage in part_coverages:
        for index, node in enumerate(part_coverage):
            if node != '1':
                continue

            disconnected_coverage[index] = 'X_%d' % (nonterminal_index,)

        nonterminal_index += 1

    return tuple(disconnected_coverage)


def compute_part_coverages(sentence, disconnected_graph_parts):
    """
    Construct disconnected graph part coverages, including coverage of the main graph part.
    :param sentence: Sentence object
    :param disconnected_graph_parts: List of disconnected graph parts, formed as sets of Node objects
    :return: List of part coverage tuples, sorted from most covering to least
    """

    part_coverages = []

    reference_coverage = sorted([node.node_id for node in sentence.source.graph.nodes])

    main_part_coverage = ['1'] * len(reference_coverage)

    for disconnected_graph_part in disconnected_graph_parts:

        part_coverage = ['0'] * len(reference_coverage)

        for node in disconnected_graph_part:
            node_index = reference_coverage.index(node.node_id)
            part_coverage[node_index] = '1'

            assert main_part_coverage[node_index] == '1'  # Make sure part coverages are non-overlapping
            main_part_coverage[node_index] = '0'

        part_coverages.append(part_coverage)

    part_coverages.append(main_part_coverage)

    # Sort part coverages from most covering to least
    part_coverages = sorted(part_coverages, key=lambda x: sum(1 for node in x if node == '1'), reverse=True)

    return part_coverages
