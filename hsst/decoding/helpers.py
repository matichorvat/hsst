import os
import time
import logging

from hsst.utility import timeout as to
from hsst.utility import utility, representation


def coverage_to_bit_string(coverage):
    return ''.join(['0' if x == 0 else '1' for x in coverage])


def coverage_to_int_string(coverage):
    return bit_coverage_to_int(coverage_to_bit_string(coverage))


def nt_coverage_to_bit_string(coverage, nonterminal):
    return ''.join(['1' if x == nonterminal else '0' for x in coverage])


def nt_coverage_to_int(coverage, nonterminal):
    return bit_coverage_to_int(nt_coverage_to_bit_string(coverage, nonterminal))


def int_to_bit_coverage(int_label):
    length = int(str(int_label[1:-7]))
    vector = int(str(int_label[-7:]))

    return '{0:b}'.format(vector).zfill(length)


def bit_coverage_to_int(bit_coverage):
    return '1%02d%07d' % (len(bit_coverage), int(bit_coverage, 2))


def compute_hsst_features(feature_dict, target_side=None):
    """
    Compute HSST rule features
    :param feature_dict: Rule's feature dictionary as provided by the grammar
    :param target_side: Option target side string for adding word penalty count
    :return: Adjusted rule's feature dictionary
    """

    adjusted_rule_feature_dict = {}

    # Add feature that this rule is hsst rule - rule penalty
    adjusted_rule_feature_dict['hsst_rule'] = -1

    # Add word penalty feature in alilats2splats
    if target_side is not None:
        target_tokens = target_side.split('_')
        target_terminal_token_len = len([x for x in target_tokens if not x.startswith('X') and not x == '999999999'])
        word_penalty_feature = - target_terminal_token_len
        adjusted_rule_feature_dict['word_insertion_penalty'] = word_penalty_feature

    if feature_dict.get('rule_type') == 'glue':
        adjusted_rule_feature_dict['hsst_glue_rule'] = -1
        return adjusted_rule_feature_dict

    elif feature_dict.get('rule_type') == 'carg_terminal':
        adjusted_rule_feature_dict['hsst_carg_terminal_rule'] = -1
        return adjusted_rule_feature_dict

    elif feature_dict.get('rule_type') == 'carg_nonterminal':
        adjusted_rule_feature_dict['hsst_carg_nonterminal_rule'] = -1
        return adjusted_rule_feature_dict

    elif feature_dict.get('rule_type') == 'disc':
        adjusted_rule_feature_dict['hsst_disc_rule'] = -1
        return adjusted_rule_feature_dict

    elif feature_dict.get('rule_type') == 'mapping_terminal':
        adjusted_rule_feature_dict['hsst_mapping_terminal_rule'] = -1
        return adjusted_rule_feature_dict

    elif feature_dict.get('rule_type') == 'mapping_nonterminal':
        adjusted_rule_feature_dict['hsst_mapping_nonterminal_rule'] = -1
        return adjusted_rule_feature_dict

    if feature_dict.get('rule_type') is not None:
        rule_type = feature_dict.get('rule_type')
        adjusted_rule_feature_dict['rt_%s' % rule_type] = -1

    # Rename probability features - they are already in -log space, so they shouldn't have - sign
    adjusted_rule_feature_dict['source2target_probability'] = feature_dict['s2t_prob']
    adjusted_rule_feature_dict['target2source_probability'] = feature_dict['t2s_prob']

    # Add count feature
    if feature_dict['count'] == 1:
        adjusted_rule_feature_dict['rule_count_1'] = -1
    elif feature_dict['count'] == 2:
        adjusted_rule_feature_dict['rule_count_2'] = -1
    else:
        adjusted_rule_feature_dict['rule_count_greater_than_2'] = -1

    return adjusted_rule_feature_dict


def order_hsst_features(feature_dict, ordered_hsst_feature_weights):
    feature_names = ordered_hsst_feature_weights.keys()
    return sorted(
        (feature_names.index(feature_name), feature_value) for feature_name, feature_value in feature_dict.items()
        if feature_name in ordered_hsst_feature_weights
    )


def loglinear_rule_weight(feature_dict, feature_weights_dict):
    """
    Compute log linear feature weight of a rule by summing the products of feature values and feature weights.
    :param feature_dict: Dictionary of features and their values
    :param feature_weights_dict: Dictionary of features and their weights
    :return: Rule weight
    """

    rule_weight = 0.0

    # Only the features present in both dictionaries affect the score
    for feature_name in feature_dict:
        if feature_name in feature_weights_dict:
            rule_weight += feature_dict[feature_name] * feature_weights_dict[feature_name]

    return rule_weight


def create_lattice(mapping, top_rtn_label, openfst):
    """
    Create sentence FST (lattice) from cell FSTs using replace operation
    :param mapping: Dictionary of labels and associated FSTs
    :param top_rtn_label: Label of the top cell in the grid
    :param openfst: OpenFST utility functions object
    :return: Sentence FST (lattice)
    """

    start_time = time.time()

    # Create root FST
    root_fst = openfst.create_root_fst(top_rtn_label, mapping)

    logging.debug('Creating FST using replace operation using %d RTNs.' % len(mapping))

    # Perform an FST replace operation to recursively replace nonterminal arc labels with cell FSTs
    lattice = openfst.replace(root_fst, mapping, epsilon=True)

    # Remove symbol tables from FSA
    lattice.isyms = None
    lattice.osyms = None

    logging.debug('FST created in %.3f seconds.' % (time.time() - start_time,))

    return lattice


def top_level_pruning(input_dir, output_dir, prune_threshold, n_best, openfst, max_duration=600):
    """
    Read lattices from disk and prune + union them with n shortest paths.
    :param input_dir: Input lattice directory path
    :param output_dir: Output lattice directory path
    :param prune_threshold: Integer indicating how much to prune
    :param n_best: Integer indicating how many best outputs to store
    :param openfst: OpenFST utility functions object
    :return:
    """

    # Iterate over files in input directory
    for i in sorted(os.listdir(input_dir)):
        if not i.endswith(".fst"):
            continue

        filename_in = os.path.join(input_dir, i)
        filename_out = os.path.join(output_dir, i)

        # Read a sentence lattice back into memory
        logging.info('Reading lattice stored at %s' % (filename_in,))
        fsa = openfst.read_fst(filename_in)

        # Compute its n_best_list
        start_time = time.time()

        logging.info('FSA has %d states and %d arcs before top level pruning.' % (
            openfst.num_states(fsa),
            openfst.num_arcs(fsa)
        ))

        try:
            with to.timeout(seconds=max_duration):
                fsa = fst_pruning(fsa, prune_threshold, n_best, openfst)

        except to.TimeoutError:
            logging.warn(
                'Top-level pruning of FSA %s was terminated due to exceeding the timeout limit of %d seconds' %
                (filename_in, max_duration))

            reduced_n_best = int(n_best/2.0)
            logging.info('Performing TLP with reduced n_best size of %d.' % (reduced_n_best,))
            fsa = fst_pruning(fsa, prune_threshold, reduced_n_best, openfst)

        logging.info('FSA has %d states and %d arcs after top level pruning, performed in %.3f seconds.' % (
            openfst.num_states(fsa),
            openfst.num_arcs(fsa),
            time.time() - start_time
        ))

        logging.info('Writing lattice to disk at %s' % (filename_out,))
        openfst.write_fst(fsa, filename_out)


def fst_pruning(fsa, prune_threshold, n_best, openfst):
    """
    Apply pruning to a single FST. If using pruning, FST is modified.
    :param fst: FST object
    :param prune_threshold: Integer indicating how much to prune
    :param n_best: Integer indicating how many best outputs to keep
    :param openfst: OpenFST utility functions object
    :return: Pruned FSA object
    """

    if prune_threshold is not None and n_best is not None:
        logging.info('Starting prune+shortest path operation.')
        openfst.prune_with_shortest_path(fsa, prune_threshold, n_best)
    elif prune_threshold is not None:
        logging.info('Starting prune path operation.')
        openfst.prune(fsa, prune_threshold)
    elif n_best is not None:
        logging.info('Starting shortest path operation.')
        fsa = openfst.shortest_path(fsa, n_best)

    return fsa


def n_best_from_disk(input_dir, result_sentences, sentence_id_map, n_best, openfst):
    """
    Read lattices from disk and store their n-best lists.
    :param input_dir: Input lattice directory path
    :param result_sentences: Dataset object storing decoded sentences
    :param sentence_id_map: Dictionary for tracking decoded sentence ids for applying LM
    :param n_best: Integer indicating how many best outputs to store
    :param openfst: OpenFST utility functions object
    :return:
    """

    # Inverse the sentence id mapping
    inv_id_map = {v: k for k, v in sentence_id_map.items()}

    # Iterate over files in input directory
    for i in os.listdir(input_dir):
        if not i.endswith(".fst"):
            continue

        filename = os.path.join(input_dir, i)
        sentence_id = inv_id_map[i.split('.')[0]]

        # Read a sentence lattice back into memory
        logging.info('Reading sentence id %s lattice stored at %s' % (sentence_id, filename))
        lattice = openfst.read_fst(filename)

        # Compute its n_best_list
        current_time = time.time()
        n_best_list = openfst.create_n_best_list(lattice, n_best, output=False)

        logging.debug('N-best list of %d items computed in %.3f seconds:' % (len(n_best_list), time.time() - current_time))

        for item in n_best_list:
            logging.debug(item)

        # Store the n_best_list in the dataset
        result_sentences.find(sentence_id).result.n_best_list = n_best_list


def store_1_best(dataset, filename, idx=True):
    """
    Store 1-best result to file.
    :param dataset: Collection of sentences with n-best lists
    :param filename: Output filename
    :param idx: Whether to store idx output or plaintext
    :return:
    """

    utility.make_sure_path_exists(os.path.dirname(filename))

    if idx:
        format='idx'
    else:
        format='text'

    with open(filename, 'wb') as rf:
        representation.result_dump(dataset, rf, format=format, encode=True)


def store_rtn(rtn, rtn_dir, int_coverage, sentence_id, openfst):
    """
    Write RTN to disk.
    :param rtn: RTN FST object
    :param rtn_dir: Path to RTN directory
    :param int_coverage: Integer coverage string
    :param sentence_id: Sentence id string
    :param openfst: OpenFST utility functions object
    :return:
    """

    # Create path for RTN
    rtn_path_dir = os.path.join(rtn_dir, sentence_id)
    rtn_path = os.path.join(rtn_path_dir, '%s.fst' % int_coverage)

    logging.debug('Writing RTN with int bit coverage %s to %s' % (int_coverage, rtn_path))

    # Store RTN to disk
    utility.make_sure_path_exists(rtn_path_dir)
    openfst.store_lattice(rtn, rtn_path, gzip=False)


def store_sentence_lattice_to_disk(sentence_lattice, sentence, sentence_id_map, outputdir, openfst):
    """
    Store sentence lattice to disk with a mapped name to support applylm tool.
    :param sentence_lattice: FSA object
    :param sentence: Sentence object
    :param sentence_id_map: Dictionary mapping original sentence ids to new sequential ids
    :param outputdir: Output directory path
    :param openfst: Openfst object
    :return:
    """
    # Map sentence id to a successive number as a range is required by the apply_lm tool
    sentence_id_map[sentence.id] = str(len(sentence_id_map))

    # Store lattice to disk
    start_time = time.time()
    lattice_path = os.path.join(outputdir, '%s.fst' % sentence_id_map[sentence.id])
    openfst.store_lattice(sentence_lattice, lattice_path, gzip=True)

    logging.info('Lattice stored to disk as %s in %.3f seconds' % (lattice_path, time.time() - start_time))
