#!/home/blue1/mh693/hsst/venv/bin/python

import os
import argparse
from collections import defaultdict

from hsst.decoding import helpers
from hsst.decoding.translation.TranslationOpenFST import TranslationOpenFST
from hsst.utility.utility import make_sure_path_exists, load_dataset

MAX_GRAPH_SIZE = 23
symbol_map = {'001': 'S',
              '002': 'D',
              '003': 'X',
              '004': 'V'}


def integrate_sentence_rtns(integrate_spaces, rtn_hiero, rtn_hsst, sentence, openfst):

    hiero_cells, top_hiero_cell_label = construct_hiero_space(rtn_hiero, openfst)
    hsst_cells, top_hsst_cell_label = construct_hsst_space(rtn_hsst, openfst)

    coverage_cells, top_cell_label, space_intersections = integrate_spaces(hiero_cells, hsst_cells, top_hiero_cell_label,
                                                                           top_hsst_cell_label, sentence, openfst)

    lattice = create_lattice(coverage_cells, top_cell_label, openfst)
    # lattice = openfst.add_start_and_end_of_sentence_symbols(lattice)

    lattice_size = sum(1 for _ in lattice.states) if lattice is not None else 0
    print 'Lattice size', lattice_size

    return lattice


def construct_hiero_space(sentence_rtn_dir, openfst):
    coverage_cells = dict()

    # Construct pathnames for sentence RTNs
    rtn_filenames = os.listdir(sentence_rtn_dir)
    rtn_paths = [(rtn_f.split('.')[0], os.path.join(sentence_rtn_dir, rtn_f)) for rtn_f in rtn_filenames]

    top_cell_label = None

    # Read RTNs into memory and add them to coverage_cells
    for label, rtn_filename in rtn_paths:
        rtn = openfst.read_fst(rtn_filename)

        # Label is hiero style
        if label.startswith('100'):
            coverage_cells[label] = rtn

        # Label is hsst style - convert from int to bit coverage
        else:
            raise Exception()
        #    bit_vector = helpers.int_to_bit_coverage(label)
        #    coverage_cells[bit_vector] = rtn

        # Switch top_cell_label if RTN label is symbol S starting at index 0
        # and covers more than previous best
        if label.startswith('1001000') and (top_cell_label is None or int(label) > int(top_cell_label)):
            top_cell_label = label

    return coverage_cells, top_cell_label


def construct_hsst_space(sentence_rtn_dir, openfst):
    coverage_cells = dict()

    # Construct pathnames for sentence RTNs
    rtn_filenames = os.listdir(sentence_rtn_dir)
    rtn_paths = [(rtn_f.split('.')[0], os.path.join(sentence_rtn_dir, rtn_f)) for rtn_f in rtn_filenames]

    highest_bit_coverage = None
    highest_bit_coverage_sum = 0

    # Read RTNs into memory and add them to coverage_cells
    for bit_vector, rtn_filename in rtn_paths:
        rtn = openfst.read_fst(rtn_filename)
        #bit_vector = helpers.int_to_bit_coverage(label)

        # Store the cell fst in coverage_cells
        coverage_cells[bit_vector] = rtn

        # Compute bit coverage sum
        bit_coverage_sum = sum([int(x) for x in list(bit_vector)])

        if bit_coverage_sum is not None and bit_coverage_sum > highest_bit_coverage_sum:
            highest_bit_coverage = bit_vector
            highest_bit_coverage_sum = bit_coverage_sum

    return coverage_cells, highest_bit_coverage


def hiero_only(hiero_cells, hsst_cells, top_hiero_cell_label, top_hsst_cell_label, sentence, openfst):

    coverage_cells = dict(hsst_cells)
    space_intersections = defaultdict(list)

    for hiero_cell_label, hiero_cell in hiero_cells.items():
        coverage_cells[hiero_cell_label] = hiero_cell

    return coverage_cells, top_hiero_cell_label, space_intersections


def add_hsst(hiero_cells, hsst_cells, top_hiero_cell_label, top_hsst_cell_label, sentence, openfst):

    coverage_cells = dict(hsst_cells)

    # Obtain alignment dictionaries from DMRS
    token_to_node_full, token_to_node_heuristic, node_to_token, reference_coverage = get_alignment_dicts(sentence.source.dmrs)
    print token_to_node_full
    print token_to_node_heuristic
    print node_to_token
    print reference_coverage

    space_intersections = defaultdict(list)

    for hiero_cell_label, hiero_cell in hiero_cells.items():
        new_label = label_convert(hiero_cell_label, token_to_node_full, node_to_token, reference_coverage)

        if new_label is not None and new_label in hsst_cells:
            print new_label
            make_connection(hiero_cell, new_label, openfst)
            space_intersections[new_label].append(hiero_cell_label)
        else:
            new_label_heur = label_convert(hiero_cell_label, token_to_node_heuristic, node_to_token, reference_coverage)

            if new_label_heur is not None and new_label_heur in hsst_cells:
                print new_label_heur
                make_connection(hiero_cell, new_label_heur, openfst)
                space_intersections[new_label_heur].append(hiero_cell_label)

        coverage_cells[hiero_cell_label] = hiero_cell

    print space_intersections
    return coverage_cells, top_hiero_cell_label, space_intersections


def add_hsst_and_back(hiero_cells, hsst_cells, top_hiero_cell_label, top_hsst_cell_label, sentence, openfst):

    coverage_cells, top_hiero_cell_label, space_intersections = add_hsst(hiero_cells, hsst_cells, top_hiero_cell_label,
                                                                         top_hsst_cell_label, sentence, openfst)

    for hsst_label, hsst_cell in hsst_cells.items():
        print 'In cell', hsst_label
        rtn_arc_join(hsst_cell, space_intersections)

    return coverage_cells, top_hiero_cell_label, space_intersections


def get_alignment_dicts(dmrs):
    """
    Get bidirectional alignment dictionaries between nodes and tokens for this sentence.
    """

    token_to_node_heuristic = defaultdict(list)
    token_to_node_full = defaultdict(list)
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
                token_to_node_full[token].append(node_id)

                # heuristic - only add to token alignments if it is not a gpred node
                if element.attrib.get('label').startswith('_'):
                    token_to_node_heuristic[token].append(node_id)

    return token_to_node_full, token_to_node_heuristic, node_to_token, sorted(reference_coverage)


def label_convert(nonterminal_label, token_to_node, node_to_token, reference_coverage):

    # Decipher non-terminal label from integer
    symbol, start, end = nonterminal_label_decipher(nonterminal_label)

    # Check alignment consistency
    if alignments_consistent(start, end, token_to_node, node_to_token):
        bit_coverage = range_to_bit_coverage(start, end, token_to_node, reference_coverage)
        new_label = helpers.bit_coverage_to_int(bit_coverage)

        print nonterminal_label, symbol, start, end, new_label, bit_coverage

        return new_label
    else:
        return None


def nonterminal_label_decipher(label):
    symbol_int = label[1:4]

    if symbol_int not in symbol_map:
        raise Exception("Unknown symbol '%s'." % symbol_int)

    symbol = symbol_map[symbol_int]

    # Account for the start of sentence symbol
    start = int(label[4:7]) - 1
    end = start + int(label[7:10])

    # Convert to bit coverage
    return symbol, start, end


def alignments_consistent(start, end, token_to_node, node_to_token):
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


def range_to_bit_coverage(start, end, token_to_node, reference_coverage):
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


def make_connection(hiero_cell, hsst_label, openfst):
    connection_fst = openfst.create_rule_fst([hsst_label], {}, {}, {})
    openfst.union(hiero_cell, connection_fst)


def create_lattice(coverage_cells, top_cell_label, openfst):

    # Create root FST
    root_fst = openfst.create_root_fst(top_cell_label, coverage_cells)

    print 'Creating cell FST using replace operation.'

    # Perform an FST replace operation to recursively replace nonterminal arc labels with cell FSTs
    lattice = openfst.replace(root_fst, coverage_cells, epsilon=True)

    # Remove epsilons and top sort the resulting FST
    lattice = openfst.rmep_min_det(lattice, min_det=True)
    openfst.top_sort(lattice)

    return lattice


def check_size(dmrs, max_size):
    return len([1 for x in dmrs if x.tag == 'node']) <= max_size


def rtn_arc_join(rtn, mapping):
    """
    Convert the non-terminal transitions of the RTN if there is a mapping defined for them.
    """

    for state in rtn.states:
        for arc in state.arcs:
            nonterminal_label = str(arc.ilabel)

            # If the edge is not a non-terminal transition, proceed to next one
            if nonterminal_label not in mapping:
                continue

            new_label = sorted(mapping[nonterminal_label])[0] # ordering by cell height not happening correctly

            print 'rewriting arc %s as %s' % (nonterminal_label, new_label)

            arc.ilabel = int(new_label)
            if nonterminal_label == str(arc.olabel):
                arc.olabel = int(new_label)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Integrate Hiero and HSST RTNs to create joint lattice.')
    parser.add_argument('-ra', '--range')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-i', '--integrate_function', choices=['hiero_only', 'add_hsst', 'add_hsst_and_back'], default='add_hsst')
    parser.add_argument('dataset')
    parser.add_argument('rtn_hiero_dir')
    parser.add_argument('rtn_hsst_dir')
    parser.add_argument('out_dir')

    args = parser.parse_args()

    # Obtain a list of sentences to operate on (intersection of hsst and hiero RTN dirs)
    rtn_subdirs = set(os.listdir(args.rtn_hiero_dir)) & set(os.listdir(args.rtn_hsst_dir))

    # Set up a range if working on a subset
    subset_range = None
    if args.range:
        try:
            # Subset range needs to start with index 0, but directories start with 1
            subset_range = (int(args.range.split('-')[0]), int(args.range.split('-')[1]) + 1)
            subset = set(range(subset_range[0], subset_range[1]))
            rtn_subdirs = filter(lambda x: int(x) in subset, rtn_subdirs)

        except (IndexError, ValueError):
            raise Exception('Incorrect specification of range. Needs to have form start-end.')

        print 'Converting a subset of %d RTNs.' % len(rtn_subdirs)

    if args.integrate_function == 'add_hsst':
        integrate_spaces_func = add_hsst
    elif args.integrate_function == 'add_hsst_and_back':
        integrate_spaces_func = add_hsst_and_back
    elif args.integrate_function == 'hiero_only':
        integrate_spaces_func = hiero_only
    else:
        raise NotImplementedError()

    # Load dataset
    dataset = load_dataset(args.dataset, subset_range=subset_range)

    # Create a sorted list of sentence rtns to process
    rtn_subdirs = [
        (sentence_id, os.path.join(args.rtn_hiero_dir, sentence_id), os.path.join(args.rtn_hsst_dir, sentence_id)) for
        sentence_id in sorted(rtn_subdirs, key=lambda x: int(x))]

    # Create output dir
    make_sure_path_exists(args.out_dir)

    openfst = TranslationOpenFST(None, None)

    for sentence_id, rtn_hiero, rtn_hsst in rtn_subdirs:
        sentence = dataset.find(sentence_id)

        # Only rewrite rtns if the number of nodes in the graph is less or equal to MAX_GRAPH_SIZE
        # This is because the non-terminal label encoding can't handle graphs larger than that
        if check_size(sentence.source.dmrs, MAX_GRAPH_SIZE):
            print 'Integrating RTNs for sentence %s.' % sentence_id
            lattice = integrate_sentence_rtns(integrate_spaces_func, rtn_hiero, rtn_hsst, sentence, openfst)
            lattice_path = os.path.join(args.out_dir, '%s.fst' % sentence_id)
            openfst.store_lattice(lattice, lattice_path)

