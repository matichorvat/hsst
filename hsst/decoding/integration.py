import os
import logging
from collections import defaultdict

from hsst.decoding import helpers
from hsst.decoding.cell_selection import Rule


def construct_hiero_space(sentence_rtn_dir, openfst):
    """
    Construct Hiero search space by loading Hiero RTNs from disk.
    :param sentence_rtn_dir: Path to Hiero RTN directory for the current sentence
    :param openfst: OpenFST utility functions object
    :return: A tuple consisting of coverage cells and HiFST top cell label
    """

    # Construct pathnames for sentence RTNs
    rtn_filenames = os.listdir(sentence_rtn_dir)
    rtn_paths = [(rtn_f.split('.')[0], os.path.join(sentence_rtn_dir, rtn_f)) for rtn_f in rtn_filenames]

    coverage_cells = dict()
    top_cell_label = None

    # Read RTNs into memory and add them to coverage_cells
    for label, rtn_filename in rtn_paths:

        if not label.startswith('100'):
            # Label is not hiero style
            continue

        rtn = openfst.read_fst(rtn_filename)
        coverage_cells[label] = rtn

        # Switch top_cell_label if RTN label is symbol S starting at index 0
        # and covers more than previous best
        if label.startswith('1001000') and (top_cell_label is None or int(label) > int(top_cell_label)):
            top_cell_label = label

    return coverage_cells, top_cell_label


def simple_hiero_into_hsst_integration(sentence, derivation_tree, hiero_cells, heuristic_intersection=False):
    # Compute space intersections
    space_intersections = hiero_hsst_space_intersection(
        sentence,
        hiero_cells.keys(),
        derivation_tree.hsst_cells.keys(),
        heuristic_intersection=heuristic_intersection
    )
    derivation_tree.space_intersections = space_intersections

    logging.debug('%d space intersections between Hiero and HSST found.' % (len(space_intersections),))

    for hsst_int_coverage in sorted(space_intersections, key=lambda x: int(x)):
        intersected_cell = derivation_tree[hsst_int_coverage]
        hiero_label_list = space_intersections[hsst_int_coverage]

        intersection_hiero_label_list = []
        if len(hiero_label_list) == 1:
            intersection_hiero_label_list = hiero_label_list

        # If an HSST RTN intersects with two Hiero RTNs, the expected case is that they cover the same span but have
        # X and V symbols. Both could be integrated, but for efficiency, if X RTN contains an arc pointing to V RTN,
        # only X RTN is integrated.
        elif len(hiero_label_list) == 2:

            # Get hiero V and X labels
            v_hiero_label, x_hiero_label = get_v_and_x_hiero_labels(hiero_label_list)

            # If X RTN does not contain an arc pointing to V RTN, integrate V RTN
            if not is_nt_arc_present(hiero_cells[x_hiero_label], v_hiero_label):
                intersection_hiero_label_list.append(v_hiero_label)

            # Integrate X RTN
            intersection_hiero_label_list.append(x_hiero_label)

        else:
            raise Exception('%d hiero labels intersect with a single HSST label %s.'
                            % (len(hiero_label_list), intersected_cell.bit_coverage))

        intersected_cell.intersections = intersection_hiero_label_list

        print intersected_cell.bit_coverage, intersection_hiero_label_list


def simple_integration(space_intersections, hiero_cells, hsst_cells, top_hiero_cell_label, openfst):
    """
    Integrate Hiero and HSST cells by adding arcs to Hiero RTNs pointing to HSST RTNs when their alignments agree.
    :param space_intersections: Dictionary with HSST labels as keys and lists of Hiero labels as values.
    :param hiero_cells: Hiero labels and associated RTN FST objects
    :param hsst_cells: HSST labels and associated RTN FST objects
    :param top_hiero_cell_label: Top hiero RTN label
    :param openfst: Openfst object
    :return: Integrated labels and associated RTN FST objects, top integrated RTN label, dictionary of HSST-Hiero label intersections
    """

    logging.debug('Simple integration of Hiero and HSST spaces based on %d space intersections' % (
        len(space_intersections,)
    ))

    joint_cells = dict(hiero_cells)
    joint_cells.update(hsst_cells)

    # Integrate spaces by adding nonterminal arcs from Hiero RTNs to HSST RTNs
    integrate_hiero_to_hsst(space_intersections, joint_cells, openfst)

    return joint_cells, top_hiero_cell_label


def extended_integration(space_intersections, hiero_cells, hsst_cells, top_hiero_cell_label, openfst,
                         established_intersections=None):
    """
    Integrate Hiero and HSST cells by both:
        - adding arcs to Hiero RTNs to point to HSST RTNs
        - rewriting HSST arcs to point to Hiero RTNs
    when their alignments agree.
    :param space_intersections: Dictionary with HSST labels as keys and lists of Hiero labels as values.
    :param hiero_cells: Hiero labels and associated RTN FST objects
    :param hsst_cells: HSST labels and associated RTN FST objects
    :param top_hiero_cell_label: Top hiero RTN label
    :param openfst: Openfst object
    :param established_intersections: Dictionary of established intersection points
    :return: Integrated labels and associated RTN FST objects, top integrated RTN label, dictionary of HSST-Hiero
    label intersections
    """

    logging.debug('Extended integration of Hiero and HSST spaces based on %d space intersections' % (
        len(space_intersections,)
    ))

    joint_cells = dict(hiero_cells)
    joint_cells.update(hsst_cells)

    if len(space_intersections) > 0:
        # Integrate spaces by adding nonterminal arcs from Hiero RTNs to HSST RTNs
        integrate_hiero_to_hsst(space_intersections, joint_cells, openfst, established_intersections)

        # Integrate spaces by adding nonterminal arcs from HSST RTNs to Hiero RTNs
        integrate_hsst_to_hiero(space_intersections, joint_cells, hsst_cells.keys())

    return joint_cells, top_hiero_cell_label


def hiero_to_hsst_label_map(sentence, hiero_cell_labels, heuristic_intersection=False):
    """
    Map Hiero labels to HSST labels when possible.
    :param sentence: Sentence object
    :param hiero_cell_labels: List of Hiero labels
    :param heuristic_intersection: Whether to ignore gpred alignment when computing HSST labels
    :return: Dictionary mapping hiero labels to HSST labels
    """

    # Obtain alignment dictionaries from node-token alignments in DMRS
    token_to_node, node_to_token, reference_coverage = bidirectional_alignment(sentence.source.dmrs)

    # If heuristic integration is enables, obtain heuristic token to node alignment dictionary
    token_to_node_heur = None

    if heuristic_intersection:
        token_to_node_heur, _, _ = bidirectional_alignment(sentence.source.dmrs, heuristic=True)

    # Determine RTN cells where Hiero and HSST alignments agree
    label_map = dict()

    for hiero_cell_label in hiero_cell_labels:

        # Attempt to construct HSST label for the given hiero span
        hsst_label = hiero_label_convert(hiero_cell_label, token_to_node, node_to_token, reference_coverage)

        if hsst_label is not None:
            label_map[hiero_cell_label] = hsst_label

        elif token_to_node_heur is not None:
            # Attempt to construct HSST label for the given hiero span using heuristic alignment
            hsst_label_heur = hiero_label_convert(hiero_cell_label, token_to_node_heur, node_to_token,
                                                  reference_coverage)

            if hsst_label_heur is not None:
                label_map[hiero_cell_label] = hsst_label_heur

    return label_map


def hiero_hsst_space_intersection(sentence, hiero_cell_labels, hsst_cell_labels, heuristic_intersection=False):
    """
    Determine RTN label pairs (HSST, Hiero) which cover/span the same tokens and can be joined. Multiple Hiero RTNs can
     intersect with a single HSST RTN.
    :param sentence: Sentence object
    :param hiero_cell_labels: List of Hiero labels
    :param hsst_cell_labels: List of HSST labels
    :param heuristic_intersection: Whether to ignore gpred alignment when computing intersection
    :return: Dictionary with HSST labels as keys and lists of Hiero labels as values.
    """

    label_map = hiero_to_hsst_label_map(sentence, hiero_cell_labels, heuristic_intersection)

    # Determine RTN cells where Hiero and HSST alignments agree
    space_intersections = defaultdict(list)

    for hiero_cell_label in hiero_cell_labels:

        if hiero_cell_label in label_map and label_map[hiero_cell_label] in hsst_cell_labels:
            space_intersections[label_map[hiero_cell_label]].append(hiero_cell_label)

    return space_intersections


def hiero_peepholes(sentence, hiero_cell_labels, heuristic_intersection=False):
    """
    Determine HSST labels of Hiero cells when possible by mapping it via alignment.
    :param sentence: Sentence object
    :param hiero_cell_labels: Dictionary of hiero labels to hiero RTNs
    :param heuristic_intersection: Whether to ignore gpred alignment when computing intersection
    :return: ionary with HSST labels as keys and lists of Hiero labels as values.
    """

    label_map = hiero_to_hsst_label_map(sentence, hiero_cell_labels, heuristic_intersection)

    hiero_peepholes = dict()
    for hiero_cell_label in hiero_cell_labels:

        if hiero_cell_label in label_map:
            hiero_peepholes[label_map[hiero_cell_label]] = hiero_cell_label

    return hiero_peepholes


def integrate_hiero_to_hsst(space_intersections, joint_cells, openfst, established_intersections=None):
    """
    Integrate at intersection points by adding nonterminal arcs to Hiero RTNs pointing to HSST RTNs.
    :param space_intersections: Dictionary of HSST labels and lists of associated hiero labels
    :param joint_cells: Dictionary of both Hiero and HSST RTN FSTs with their labels as keys
    :param openfst: Openfst object
    :param established_intersections: Dictionary of established intersection points
    :return:
    """

    # Only process intersections not previously established
    if established_intersections is not None:
        space_intersections = dict(x for x in space_intersections.items() if
                                   x not in established_intersections.items())
        established_intersections.update(space_intersections)

    for hsst_label, hiero_label_list in space_intersections.items():

        # If an HSST RTN intersects with a single Hiero RTN, integrate them together
        if len(hiero_label_list) == 1:
            hiero_label = hiero_label_list[0]
            integrate_hiero_to_hsst_rtn(joint_cells[hiero_label], hsst_label, openfst)

        # If an HSST RTN intersects with two Hiero RTNs, the expected case is that they cover the same span but have
        # X and V symbols. Both could be integrated, but for efficiency, if X RTN contains an arc pointing to V RTN,
        # only V RTN is integrated.
        elif len(hiero_label_list) == 2:

            # Get hiero V and X labels
            v_hiero_label, x_hiero_label = get_v_and_x_hiero_labels(hiero_label_list)

            # If X RTN does not contain an arc pointing to V RTN, integrate X RTN
            if not is_nt_arc_present(joint_cells[x_hiero_label], v_hiero_label):
                integrate_hiero_to_hsst_rtn(joint_cells[x_hiero_label], hsst_label, openfst)

            # Integrate V RTN
            integrate_hiero_to_hsst_rtn(joint_cells[v_hiero_label], hsst_label, openfst)

        else:
            raise Exception('%d hiero labels intersect with a single HSST label %s.'
                            % (len(hiero_label_list), hsst_label))


def integrate_hsst_to_hiero(space_intersections, joint_cells, hsst_labels):
    """
    Integrate at intersection points by changing nonterminal arcs from pointing to HSST RTNs to point to Hiero RTNs.
    :param space_intersections: Dictionary of HSST labels and lists of associated hiero labels
    :param joint_cells: Dictionary of both Hiero and HSST RTN FSTs with their labels as keys
    :param hsst_labels: List of HSST RTN labels
    """

    for hsst_label in hsst_labels:
        hsst_cell = joint_cells[hsst_label]

        for state in hsst_cell.states:
            for arc in state.arcs:

                hsst_label = str(arc.olabel)  # Needs to be output label so that alignment transducers also get matched

                # If the arc is not a non-terminal transition that we can integrate, proceed to the next one
                if hsst_label not in space_intersections:
                    continue

                hiero_label_list = space_intersections[hsst_label]

                if len(hiero_label_list) == 1:
                    hiero_label = hiero_label_list[0]

                    integrate_hsst_to_hiero_rtn_rewrite_arc(arc, hsst_label, hiero_label)

                # If an HSST RTN intersects with two Hiero RTNs, the expected case is that they cover the same span but
                # have X and V symbols. Both could be integrated, but for efficiency, if X RTN contains an arc pointing
                # to V RTN, only X RTN is integrated.
                elif len(hiero_label_list) == 2:

                    # Get hiero V and X labels
                    v_hiero_label, x_hiero_label = get_v_and_x_hiero_labels(hiero_label_list)

                    # If X RTN does not contain an arc pointing to V RTN, integrate V RTN
                    if not is_nt_arc_present(joint_cells[x_hiero_label], v_hiero_label):
                        integrate_hsst_to_hiero_rtn_rewrite_arc(arc, hsst_label, v_hiero_label)

                    # Integrate X RTN
                    integrate_hsst_to_hiero_rtn_rewrite_arc(arc, hsst_label, x_hiero_label)

                else:
                    raise Exception('%d hiero labels intersect with a single HSST label %s.'
                                    % (len(hiero_label_list), hsst_label))


def bidirectional_alignment(dmrs, heuristic=False):
    """
    Get bidirectional alignment dictionaries between nodes and tokens for this sentence and the reference coverage.
    :param dmrs: DMRS xml object
    :params heuristic: If true, omit alignments with gpred nodes
    :return: Token to node and node to token dictionaries, and reference coverage list
    """

    token_to_node = defaultdict(list)
    node_to_token = dict()
    reference_coverage = list()

    for element in dmrs:

        if element.tag == 'node':
            node_id = element.attrib.get('nodeid')
            tokalign = element.attrib.get('tokalign')

            reference_coverage.append(node_id)

            if tokalign == '':
                continue

            tokens = [int(x) for x in tokalign.split(' ') if x != '-1']

            if len(tokens) == 0:
                continue

            node_to_token[node_id] = tokens

            for token in tokens:

                if not heuristic:
                    token_to_node[token].append(node_id)
                elif element.attrib.get('label').startswith('_'):
                    # heuristic - only add to token alignments if it is not a gpred node
                    token_to_node[token].append(node_id)

    return token_to_node, node_to_token, sorted(reference_coverage)


def hiero_label_convert(hiero_nt_label, token_to_node, node_to_token, reference_coverage):
    """
    Convert hiero nonterminal label to HSST nonterminal label if alignment is consistent, otherwise return None.
    :param hiero_nt_label: Hiero nonterminal integer label string
    :param token_to_node: Token to node alignment dictionary
    :param node_to_token: Node to token alignment dictionary
    :param reference_coverage: Reference coverage list
    :return: HSST label string or None
    """

    hsst_label = None

    # Decompose hiero non-terminal label
    symbol, start, end = hiero_nt_int_label_decompose(hiero_nt_label)

    # Check alignment consistency
    if is_alignment_consistent(start, end, token_to_node, node_to_token):
        bit_coverage = span_to_bit_coverage(start, end, token_to_node, reference_coverage)
        hsst_label = helpers.bit_coverage_to_int(bit_coverage)

    return hsst_label


hiero_symbol_map = {'001': 'S', '002': 'D', '003': 'X', '004': 'V'}


def hiero_nt_int_label_decompose(hiero_label):
    """
    Decompose hiero integer nonterminal label into parts: symbol, start, and end.
    Note that start corresponds to HSST start, i.e. <s> is at position -1.
    :param hiero_label: Hiero integer label string
    :return: Tuple of (symbol, start, end)
    """

    symbol_int = hiero_label[1:4]

    if symbol_int not in hiero_symbol_map:
        raise Exception("Unknown symbol '%s'." % symbol_int)

    symbol = hiero_symbol_map[symbol_int]

    # Account for the start of sentence symbol
    start = int(hiero_label[4:7]) - 1
    end = start + int(hiero_label[7:10])

    # Convert to bit coverage
    return symbol, start, end


def is_alignment_consistent(start, end, token_to_node, node_to_token):
    """
    Check whether alignment between token span start:end and corresponding DMRS nodes is consistent.
    :param start: Start token index
    :param end: End token index
    :param token_to_node: Token to node alignment dictionary
    :param node_to_token: Node to token alignment dictionary
    :return: Boolean
    """

    aligned_node_list = list()

    token_list = range(start, end + 1)
    for token in token_list:
        aligned_node_list.extend(token_to_node[token])

    aligned_token_list = list()
    for node in aligned_node_list:
        aligned_token_list.extend(node_to_token[node])

    if set(token_list) == set(aligned_token_list):
        return True
    else:
        return False


def span_to_bit_coverage(start, end, token_to_node, reference_coverage):
    """
    Construct a bit coverage for token span start:end according to the reference coverage.
    :param start: Start token index
    :param end: End token index
    :param token_to_node: Token to node alignment dictionary
    :param reference_coverage: Reference coverage list
    :return: Bit coverage string
    """
    aligned_node_list = list()

    token_list = range(start, end + 1)
    for token in token_list:
        aligned_node_list.extend(token_to_node[token])

    node_set = set(aligned_node_list)
    coverage = ['0'] * len(reference_coverage)

    for index, node_id in enumerate(reference_coverage):
        if node_id in node_set:
            coverage[index] = '1'

    return ''.join(coverage)


def get_v_and_x_hiero_labels(hiero_label_list):
    """
    Make sure and select hiero labels from hiero_label_list that have V and X symbols.
    :param hiero_label_list: A list of 2 hiero labels
    :return: Hiero label with symbol V, Hiero label with symbol X
    """

    hiero_label_1 = hiero_label_list[0]
    hiero_label_2 = hiero_label_list[1]

    symbol_1, start_1, end_1 = hiero_nt_int_label_decompose(hiero_label_1)
    symbol_2, start_2, end_2 = hiero_nt_int_label_decompose(hiero_label_2)

    if start_1 != start_2 or end_1 != end_2:
        raise Exception('Hiero labels %s and %s do not have the same span but have the same HSST label.'
                        % (hiero_label_1, hiero_label_2))

    if symbol_1 not in {'X', 'V'} or symbol_2 not in {'X', 'V'} or symbol_1 == symbol_2:
        raise Exception('Hiero labels %s and %s have unexpected symbol.' % (hiero_label_1, hiero_label_2))

    v_hiero_label = hiero_label_1 if symbol_1 == 'V' else hiero_label_2
    x_hiero_label = hiero_label_1 if symbol_1 == 'X' else hiero_label_2

    return v_hiero_label, x_hiero_label


def is_nt_arc_present(rtn, nt_arc_label):
    """
    Check if rtn contains nonterminal arc with label nt_arc_label.
    :param rtn: RTN FST object
    :param nt_arc_label: Nonterminal arc label
    :return: Boolean
    """

    nt_arc_label = int(nt_arc_label)

    for state in rtn.states:
        for arc in state.arcs:
            if nt_arc_label == arc.olabel:
                return True

    return False


def integrate_hiero_to_hsst_rtn(rtn, hsst_label, openfst):
    """
    Integrate Hiero RTN with HSST RTN with label hsst_label by creating a single arc FST and unioning it with Hiero RTN.
    :param rtn: RTN FST object
    :param hsst_label: HSST label string
    :param openfst: OpenFST object
    :return:
    """

    logging.debug('Adding arc from hiero RTN to HSST RTN with label %s.' % hsst_label)

    rule = Rule(0, [hsst_label], None, ())

    connection_fst = openfst.create_rule_fst(rule, {})
    openfst.union(rtn, connection_fst)


def integrate_hiero_into_hsst_rtn(rtn, hiero_label, openfst, feature_weights_dict):
    """
    Integrate Hiero RTN with HSST RTN with label hiero_label by creating a single arc FST and unioning it with HSST RTN.
    :param rtn: RTN FST object
    :param hiero_label: Hiero label string
    :param openfst: OpenFST object
    :param feature_weights_dict: Dictionary of feature names and their weights
    :return:
    """

    logging.debug('Unioning Hiero RTN with label %s with HSST RTN.' % hiero_label)

    rule = Rule(0, [hiero_label], None, (), hiero_intersection_rule=True)
    # Set hiero feature dict separately so it's not computed like for HSST
    rule.feature_dict = compute_hiero_intersection_features()

    connection_fst = openfst.create_rule_fst(rule, feature_weights_dict)
    openfst.union(rtn, connection_fst)


def integrate_hsst_to_hiero_rtn_rewrite_arc(rtn_arc, hsst_label, hiero_label):
    """
    Integrate HSST RTN with Hiero RTN by changing arc label to point to Hiero RTN label.
    :param rtn_arc: RTN FST arc
    :param hsst_label: HSST label string
    :param hiero_label: Hiero label string
    :return:
    """

    # logging.debug('Rewriting arc %s to %s.' % (hsst_label, hiero_label))

    rtn_arc.olabel = int(hiero_label)

    if hsst_label == str(rtn_arc.ilabel):
        rtn_arc.ilabel = int(hiero_label)


def compute_hiero_intersection_features():
    return {'hiero_intersec': -1}
