#!/home/blue1/mh693/hsst/venv/bin/python

import os
import shutil
import logging
import argparse
from collections import defaultdict

from hsst.decoding.translation.TranslationOpenFST import TranslationOpenFST
from hsst.utility.utility import make_sure_path_exists, load_dataset


MAX_GRAPH_SIZE = 23
symbol_map = {'001': 'S',
              '002': 'D',
              '003': 'X',
              '004': 'V'}

openfst = TranslationOpenFST(None, None)


def rewrite_sentence_rtns(sentence_rtn_dir, sentence_rnt_out_dir, sentence):
    """
    Rewrite RTNs of the current sentence
    """

    # Obtain alignment dictionaries from DMRS
    token_to_node, node_to_token, reference_coverage = get_alignment_dicts(sentence.source.dmrs)

    logging.debug(token_to_node)
    logging.debug(node_to_token)
    logging.debug(reference_coverage)

    # Storing mappings for fast retrieval
    mapping = dict()

    # Construct pathnames to all sentence RTNs
    sentence_rtn_filenames = [(rtn_filename, os.path.join(sentence_rtn_dir, rtn_filename)) for rtn_filename in
                              os.listdir(sentence_rtn_dir)]

    # Iterate over the RTNs, converting one at a time
    for rtn_name, rtn_filename in sentence_rtn_filenames:

        # Convert the RTN filename
        rtn_name = rtn_name.split('.')[0]
        rtn_name_converted = label_convert(rtn_name, mapping, token_to_node, node_to_token, reference_coverage, symbol_prefix=True)

        # Read the RTN to memory
        rtn = openfst.read_fst(rtn_filename)

        # Convert RTN
        converted_rtn = rtn_convert(rtn, token_to_node, node_to_token, reference_coverage, mapping)

        # Write it to disk
        openfst.write_fst(converted_rtn, os.path.join(sentence_rnt_out_dir, (rtn_name if rtn_name_converted is None else str(rtn_name_converted)) + '.fst'))


def check_size(dmrs, max_size):
    return len([1 for x in dmrs if x.tag == 'node']) <= max_size


def get_alignment_dicts(dmrs):
    """
    Get bidirectional alignment dictionaries between nodes and tokens for this sentence.
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
                token_to_node[token].append(node_id)

    return token_to_node, node_to_token, sorted(reference_coverage)


def rtn_convert(rtn, token_to_node, node_to_token, reference_coverage, mapping):
    '''
    Convert the non-terminal transitions of the RTN.
    '''

    # logging.debug('RTN')
    # logging.debug(openfst.fst_to_str(rtn, {}))

    for state in rtn.states:
        for arc in state.arcs:
            nonterminal_label = str(arc.ilabel)

            # If the edge is not a non-terminal transition, proceed to next one
            if not nonterminal_label_match(nonterminal_label):
                continue

            new_label = label_convert(nonterminal_label, mapping, token_to_node, node_to_token, reference_coverage)

            # If alignment isn't consistent do not convert
            if new_label is None:
                continue

            # Assign new ids to ilabel and olabel
            if nonterminal_label == str(arc.olabel):
                arc.ilabel = int(new_label)
                arc.olabel = int(new_label)

            else:
                new_label_o = label_convert(str(arc.olabel), mapping, token_to_node, node_to_token, reference_coverage)
                if new_label_o is not None:
                    arc.ilabel = int(new_label)
                    arc.olabel = int(new_label_o)

    return rtn


def nonterminal_label_match(label):
    if len(label) != 10:
        return False

    if label[0] != '1':
        return False

    return True


def label_convert(nonterminal_label, mapping, token_to_node, node_to_token, reference_coverage, symbol_prefix=False):
    # If label has previously been mapped, used that, otherwise compute the mapping

    new_label = None
    if nonterminal_label in mapping and not symbol_prefix:
        new_label = mapping[nonterminal_label]

    else:
        # Decipher non-terminal label from integer
        symbol, start, end = nonterminal_label_decipher(nonterminal_label)
        logging.debug('%s %s %s %s' %(symbol, start, end, nonterminal_label))

        # Check alignment consistency
        if alignments_consistent(start, end, token_to_node, node_to_token):
            bit_coverage = range_to_bit_coverage(start, end, token_to_node, reference_coverage)
            new_label = bit_coverage_to_int(bit_coverage)

            if symbol_prefix:
                new_label = "%s_%s" % (symbol, new_label)
            else:
                mapping[nonterminal_label] = new_label

    return new_label


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


def bit_coverage_to_int(bit_vector):
    # This has potential for clashes with the regular wmap id arcs, if max_wmap_id exceeds 10 million
    # Namely when the lenght of the sentence is 1, the int will be 1xxxxxxx
    # The alternative of putting a 1 in front of the number has potential for clashes with Hiero coding
    # Putting a 2 in front would allow for only sentences of length 14 or less
    return int('1%02d%07d' % (len(bit_vector), int(bit_vector, 2)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Convert RTNs to use bit coverage instead of word spans.')
    parser.add_argument('-ra', '--range')
    parser.add_argument('-l', '--log')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-s', '--store', help='Store (non)filtered decisions in a file')
    parser.add_argument('dataset')
    parser.add_argument('rtn_in_dir')
    parser.add_argument('rtn_out_dir')

    args = parser.parse_args()

    # Set up logging
    if args.log:
        logging.basicConfig(filename=args.log, level=logging.DEBUG if args.verbose else logging.INFO)
    else:
        logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    # Obtain a list of sentences to operate on
    rtn_subdirs = os.listdir(args.rtn_in_dir)

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

        logging.info('Converting a subset of %d RTNs.' % len(rtn_subdirs))

    # Load dataset
    dataset = load_dataset(args.dataset, subset_range=subset_range)

    # Create a sorted list of sentence rtns to process
    rtn_subdirs = [(sentence_id, os.path.join(args.rtn_in_dir, sentence_id)) for sentence_id in sorted(rtn_subdirs, key=lambda x: int(x))]

    # Create output dir
    make_sure_path_exists(args.rtn_out_dir)

    if args.store is not None:
        fstore = open(args.store, 'wb')

    for sentence_id, sentence_rtn_in_dir in rtn_subdirs:
        sentence = dataset.find(sentence_id)

        sentence_rnt_out_dir = os.path.join(args.rtn_out_dir, sentence_id)

        # Only rewrite rtns if the number of nodes in the graph is less or equal to MAX_GRAPH_SIZE
        # This is because the non-terminal label encoding can't handle graphs larger than that
        if check_size(sentence.source.dmrs, MAX_GRAPH_SIZE):
            logging.info('Converting RTNs for sentence %s.' % sentence_id)
            make_sure_path_exists(sentence_rnt_out_dir)
            rewrite_sentence_rtns(sentence_rtn_in_dir, sentence_rnt_out_dir, sentence)

            if args.store is not None:
                fstore.write('1\n')

        else:
            # Otherwise copy rtns over
            logging.info('Sentence %s too long, copying RTNs.' % sentence_id)

            if os.path.isdir(sentence_rnt_out_dir):
                logging.warn('Removing existing directory for sentence %s.' % sentence_id)
                shutil.rmtree(sentence_rnt_out_dir)

            shutil.copytree(sentence_rtn_in_dir, sentence_rnt_out_dir)

            if args.store is not None:
                fstore.write('0\n')

    if args.store is not None:
        fstore.close()

