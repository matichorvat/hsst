import os
import time
import logging

from hsst.decoding import helpers
from hsst.decoding import integration
from hsst.decoding import localpruning
from hsst.decoding.Decoder import Decoder
from hsst.utility import utility
from hsst.utility import timeout as to
from hsst.utility.representation import Result, Sentence, SentenceCollection

LAT_STORAGE_DIR = 'TMP_LATS/'
LM_LAT_STORAGE_DIR = 'LATS/'
TOP_LEVEL_PRUNING_STORAGE_DIR = 'LATS_PRUNED/'


class TranslationDecoder(Decoder):

    def initialize_tmp_dirs(self):
        """
        Creates temporary directories for storing translation and post-LM lattices.
        :return: Paths to the temporary directories
        """

        lats_dir = os.path.join(self.temporary_storage, LAT_STORAGE_DIR)
        lm_dir = os.path.join(self.temporary_storage, LM_LAT_STORAGE_DIR)

        utility.create_dir(lats_dir)
        utility.create_dir(lm_dir)

        logging.debug('Temporary directories %s and %s created.' % (lats_dir, lm_dir))

        tlp_dir = None
        if self.top_level_pruning is not None:
            tlp_dir = os.path.join(self.temporary_storage, TOP_LEVEL_PRUNING_STORAGE_DIR)
            utility.create_dir(tlp_dir)
            logging.debug('Temporary directory %s.' % (tlp_dir,))

        return lats_dir, lm_dir, tlp_dir

    def __call__(self, dataset, grammar_path, coverage_path, language_model, feature_weights_dict, openfst,
                       best1_output, outdir, hiero_dir=None, n_best=1):
        """
        Decode a dataset in translation mode.
        :param dataset: Dataset object containing sentences to decode
        :param grammar_path: Path to the sentence specific grammars
        :param coverage_path: Path to the sentence specific coverages
        :param language_model: Initialized LanguageModel object
        :param feature_weights_dict: Dictionary of feature names and their weights
        :param openfst: OpenFST utility functions object
        :param best1_output: Path to file storing 1-best output
        :param outdir: Permanent lattice storage directory path
        :param hiero_dir: Hiero RTN directory path
        :param n_best: Number of best outputs to store for each sentence
        :return:
        """

        logging.info('Starting decoding of the dataset in translation mode.')

        # Create temporary directories
        lats_dir, lm_dir, tlp_dir = self.initialize_tmp_dirs()

        # Initialize result sentence collection
        result_dataset = SentenceCollection()

        # If local pruning is enabled, load the local pruning LM from disk as an FSA
        # Warning, the FSA needs to be arc sorted
        local_pruning_lm = None
        if self.local_pruning and not self.hiero_only:
            logging.info('Loading pruning LM from disk')
            local_pruning_lm = openfst.read_fst(self.local_pruning_settings['pruning_lm'])

        # Track the decoded ids for applying LM
        sentence_id_map = dict()

        # Iterate over the sentences and attempt to decode them
        for index, sentence in enumerate(dataset):
            result_sentence = self.decode_sentence(
                index,
                sentence,
                grammar_path,
                coverage_path,
                feature_weights_dict,
                local_pruning_lm,
                openfst,
                sentence_id_map,
                lats_dir,
                hiero_dir
            )
            result_dataset.add(result_sentence)

        # If no sentences decoded successfully, terminate decoding of the dataset
        if not any(True if sentence.result is not None else False for sentence in result_dataset):
            logging.debug('No lattices produced in decoding, terminating.')
            return

        utility.set_memory_limit(-1)

        # Apply the language model to the directory of lattices
        current_time = time.time()
        logging.info('Applying the language model to dataset lattices.')

        language_model(
            lats_dir,
            lm_dir,
            (0, len(sentence_id_map) - 1),
            feature_weights_dict['language_model_probability']
        )

        logging.info('LM application finished in %.3f seconds' % (time.time() - current_time))

        if self.top_level_pruning is not None:
            # Do top level pruning to reduce the size of the lattices stored on disk
            prune_threshold, pruning_n_best = self.top_level_pruning

            logging.info('Starting top level pruning with pruning threshold %d and %d shortest paths' % (
                prune_threshold if prune_threshold is not None else 0,
                pruning_n_best
            ))

            start_time = time.time()
            helpers.top_level_pruning(lm_dir, tlp_dir, prune_threshold, pruning_n_best, openfst)
            logging.info('Top level pruning finished in %.3f seconds.' % (
                time.time() - start_time,
            ))

        if self.testing_mode:
            # Compute n-best list for each lattice and store it in the result dataset
            current_time = time.time()
            logging.info('Reading dataset lattices back into memory and computing n-best lists.')
            helpers.n_best_from_disk(lm_dir, result_dataset, sentence_id_map, n_best, openfst)

            logging.info('N-best list computation finished in %.3f seconds' % (time.time() - current_time))

            # Store 1-best results to file
            helpers.store_1_best(result_dataset, best1_output + '.txt', idx=False)

        # Store final lattices
        if self.top_level_pruning is None:
            store_and_cleanup(lm_dir, outdir, sentence_id_map, self.gzip, self.temporary_storage)
        else:
            store_and_cleanup(tlp_dir, outdir, sentence_id_map, self.gzip, self.temporary_storage)

    def decode_sentence(self, index, sentence, grammar_path, coverage_path, feature_weights_dict, local_pruning_lm,
                        openfst, sentence_id_map, lats_dir, hiero_dir):
        """
        Decode a single sentence and save its lattice to disk (without LM applied).
        :param index: Sentence index in the dataset
        :param sentence: Sentence object
        :param grammar_path: Path to the sentence specific grammars
        :param coverage_path: Path to the sentence specific coverages
        :param feature_weights_dict: Dictionary of feature names and their weights
        :param local_pruning_lm: Language model FSA for local pruning
        :param openfst: OpenFST utility functions object
        :param sentence_id_map: Dictionary for tracking decoded sentence ids for applying LM
        :param lats_dir: Directory path for storing lattices
        :param hiero_dir: Hiero RTN directory path
        :return:
        """

        logging.info('Starting decoding of sentence with index %d and id %s' % (index, sentence.id))
        start_time = time.time()

        if self.hiero_only:
            logging.info('Starting decoding of sentence %s in hiero only mode.' % (sentence.id,))
            sentence_lattice = self.hiero_only_mode(sentence, hiero_dir, openfst)

        elif self.filtering is not None and not self.filtering.filter(sentence):
            # If sentence is filtered out for HSST decoding, try to use hiero
            logging.info('Sentence %s filtered out of HSST decoding.' % (sentence.id,))

            if hiero_dir is None or not self.hiero_backoff:
                logging.info('Returning empty result for sentence %s.' % (sentence.id,))
                return Sentence(sentence.id)

            else:
                logging.info('Starting decoding of sentence %s in hiero only mode.' % (sentence.id,))
                sentence_lattice = self.hiero_only_mode(sentence, hiero_dir, openfst)

        else:

            try:
                # Load sentence specific grammar
                logging.info('Loading grammar for sentence %s from %s.' % (sentence.id, grammar_path))
                sentence_grammar = utility.load_sentence_specific_grammar(sentence.id, grammar_path)
                logging.info('Loading coverage for sentence %s from %s.' % (sentence.id, coverage_path))
                sentence_coverage = utility.load_sentence_specific_coverage(sentence.id, coverage_path)

                with to.timeout(seconds=self.timeout):
                    if hiero_dir is None:
                        logging.info('Starting decoding of sentence %s in HSST only mode.' % (sentence.id,))
                        sentence_lattice = self.hsst_only_mode(
                            sentence,
                            sentence_grammar,
                            sentence_coverage,
                            feature_weights_dict,
                            local_pruning_lm,
                            openfst
                        )

                    else:
                        logging.info('Starting decoding of sentence %s in hiero+HSST mode.' % (sentence.id,))
                        sentence_lattice = self.hsst_hiero_mode(
                            sentence,
                            sentence_grammar,
                            sentence_coverage,
                            feature_weights_dict,
                            local_pruning_lm,
                            openfst,
                            hiero_dir
                        )

            except to.TimeoutError:
                logging.warn('Decoding of sentence %s (%s) was terminated due to exceeding the timeout limit of %d seconds' % (
                    sentence.id,
                    sentence.orig_id,
                    self.timeout
                ))

                if hiero_dir is None or not self.hiero_backoff:
                    # Return empty result sentence to indicate decoding was attempted
                    logging.error('Decoding of sentence %s (%s) was terminated due to timeout and no backup possible. Sentence will not be decoded.' % (
                        sentence.id,
                        sentence.orig_id
                    ))
                    return Sentence(sentence.id)

                else:
                    # Relaunch decoding using Hiero-only mode
                    logging.info('Starting decoding of sentence %s in hiero only mode.' % (sentence.id,))
                    sentence_lattice = self.hiero_only_mode(sentence, hiero_dir, openfst)

            except MemoryError:
                logging.error(
                    'Decoding of sentence %s (%s) was terminated due to exceeding allowed memory after %d seconds' % (
                        sentence.id, sentence.orig_id, time.time() - start_time))

                if hiero_dir is None or not self.hiero_backoff:
                    logging.error('Decoding of sentence %s (%s) was terminated due to exceeding maximum memory and no backup possible. Sentence will not be decoded.' % (
                        sentence.id,
                        sentence.orig_id
                    ))
                    return Sentence(sentence.id)

                else:
                    # Relaunch decoding using Hiero-only mode
                    logging.info('Starting decoding of sentence %s in hiero only mode.' % (sentence.id,))
                    sentence_lattice = self.hiero_only_mode(sentence, hiero_dir, openfst)

        # If decoding failed, skip storing lattice to the disk
        if sentence_lattice is None:
            logging.info('Decoding of sentence with index %d and id %s failed in %.3f seconds' % (
                index, sentence.id, time.time() - start_time))

            # Return empty result sentence to indicate decoding was attempted
            return Sentence(sentence.id)

        logging.info('Decoding of sentence with index %d and id %s finished in %.3f seconds' % (
            index, sentence.id, time.time() - start_time))

        # Store sentence lattice to disk with a mapped name to support applylm tool.
        helpers.store_sentence_lattice_to_disk(sentence_lattice, sentence, sentence_id_map, lats_dir, openfst)

        # Return sentence marking it as successfully decoded
        return Sentence(sentence.id, result=Result())

    def hiero_only_mode(self, sentence, hiero_dir, openfst):
        """
        Create a translation acceptor (lattice) for the given sentence using only Hiero RTNs.
        :param sentence: Sentence object
        :param hiero_dir: Hiero RTN directory path
        :param openfst: OpenFST utility functions object
        :return:
        """

        start_time = time.time()

        # Load hiero RTNs
        hiero_subdir = os.path.join(hiero_dir, sentence.id)
        hiero_cells, top_hiero_cell_label = integration.construct_hiero_space(hiero_subdir, openfst)

        logging.debug('Constructing a lattice out of %d RTNs, top cell is %s.' % (
            len(hiero_cells),
            top_hiero_cell_label
        ))

        # Construct Hiero FSA
        sentence_lattice = helpers.create_lattice(hiero_cells, top_hiero_cell_label, openfst)

        logging.debug('Removing epsilons in sentence_lattice FST with %d states and %d arcs.' % (
            openfst.num_states(sentence_lattice),
            openfst.num_arcs(sentence_lattice))
        )

        sentence_lattice = openfst.rmep_min_det(sentence_lattice, min_det=False)

        lattice_size = openfst.num_states(sentence_lattice) if sentence_lattice is not None else 0
        logging.info('Lattice with %d states and %d arcs constructed in %.3f seconds' % (
            lattice_size,
            openfst.num_arcs(sentence_lattice),
            time.time() - start_time)
        )

        if lattice_size == 0:
            logging.error('Constructed lattice is empty. Returning None.')
            return None

        return sentence_lattice

    def hsst_only_mode(self, sentence, sentence_grammar, sentence_coverage, feature_weights_dict, local_pruning_lm,
                       openfst):
        """
        Create a translation acceptor (lattice) for the given sentence using only HSST.
        :param sentence: Sentence object
        :param sentence_grammar: Dictionary of rule_id:rule instances for the sentence
        :param sentence_coverage: Dictionary of coverages with associated rules for the sentence
        :param feature_weights_dict: Dictionary of feature names and their weights
        :param local_pruning_lm: Language model FSA for local pruning
        :param openfst: OpenFST utility functions object
        :return: Sentence FSA (lattice) object
        """

        start_time = time.time()

        # Populate the coverage cells with HSST RTNs
        derivation_tree = self.construct_hsst_space(
            sentence,
            sentence_grammar,
            sentence_coverage,
            feature_weights_dict,
            local_pruning_lm,
            openfst
        )

        if len(derivation_tree) == 0:
            logging.error('No cells constructed from derivation tree. Returning None.')
            return None

        logging.debug('Constructing a lattice out of %d cell fsts, top cell is %s (%s).' % (
            len(derivation_tree),
            derivation_tree.top_hsst_cell.bit_coverage,
            derivation_tree.top_hsst_cell.int_coverage
        ))

        # Construct HSST FSA
        sentence_lattice = helpers.create_lattice(
            dict((int_coverage, cell.rtn) for int_coverage, cell in derivation_tree.hsst_cells.items()),
            derivation_tree.top_hsst_cell.int_coverage,
            openfst
        )

        # Projecting input labels to outputs (DR symbols can't cause problems anymore)
        logging.debug('Projecting input labels')
        sentence_lattice = openfst.project(sentence_lattice, output=False)

        logging.debug('Removing epsilons in sentence_lattice FST with %d states and %d arcs.' % (
            openfst.num_states(sentence_lattice),
            openfst.num_arcs(sentence_lattice))
        )

        sentence_lattice = openfst.rmep_min_det(sentence_lattice, min_det=False)

        lattice_size = openfst.num_states(sentence_lattice) if sentence_lattice is not None else 0
        logging.info('Lattice with %d states and %d arcs constructed in %.3f seconds' % (
            lattice_size,
            openfst.num_arcs(sentence_lattice),
            time.time() - start_time)
        )

        if lattice_size == 0:
            logging.error('Constructed lattice is empty. Returning None.')
            return None

        # Add start and end of sentence symbols before LM application
        sentence_lattice = openfst.add_start_and_end_of_sentence_symbols(sentence_lattice)

        return sentence_lattice

    def hsst_hiero_mode(self, sentence, sentence_grammar, sentence_coverage, feature_weights_dict, local_pruning_lm,
                        openfst, hiero_dir):
        """
        Create a translation acceptor (lattice) for the given sentence using both hiero RTNs and HSST.
        :param sentence: Sentence object
        :param sentence_grammar: Dictionary of rule_id:rule instances for the sentence
        :param sentence_coverage: Dictionary of coverages with associated rules for the sentence
        :param feature_weights_dict: Dictionary of feature names and their weights
        :param local_pruning_lm: Language model FSA for local pruning
        :param openfst: OpenFST utility functions object
        :param hiero_dir: Hiero RTN directory path
        :return: Sentence FSA (lattice) object
        """

        start_time = time.time()

        # Load hiero RTNs
        hiero_subdir = os.path.join(hiero_dir, sentence.id)
        logging.debug('Constructing hiero space.')
        hiero_cells, top_hiero_cell_label = integration.construct_hiero_space(hiero_subdir, openfst)

        # Track added hiero->hsst links if using local pruning and extended integration to prevent duplicating them
        if self.local_pruning and self.extended_integration_opt:
            established_intersections = dict()
        else:
            established_intersections = None

        # Populate the coverage cells with HSST RTNs
        derivation_tree = self.construct_hsst_space(
            sentence,
            sentence_grammar,
            sentence_coverage,
            feature_weights_dict,
            local_pruning_lm,
            openfst,
            hiero_cells=hiero_cells,
            established_intersections=established_intersections
        )

        if len(derivation_tree) == 0:
            logging.error('No cells constructed from derivation tree. Returning None.')
            return None

        # Construct HSST FSA
        top_cell = derivation_tree.top_hsst_cell

        logging.debug('Constructing a lattice out of %d cell fsts, top cell is %s (%s).' % (
            len(derivation_tree),
            top_cell.bit_coverage,
            top_cell.int_coverage
        ))

        joint_rtns = dict((int_coverage, cell.rtn) for int_coverage, cell in derivation_tree.hsst_cells.items())
        joint_rtns.update(hiero_cells)

        sentence_lattice = helpers.create_lattice(
            joint_rtns,
            top_cell.int_coverage,
            openfst
        )

        # Projecting input labels to outputs (DR symbols can't cause problems anymore)
        logging.debug('Projecting input labels')
        sentence_lattice = openfst.project(sentence_lattice, output=False)

        logging.debug('Removing epsilons in sentence_lattice FST with %d states and %d arcs.' % (
            openfst.num_states(sentence_lattice),
            openfst.num_arcs(sentence_lattice))
        )

        sentence_lattice = openfst.rmep_min_det(sentence_lattice, min_det=False)

        lattice_size = openfst.num_states(sentence_lattice) if sentence_lattice is not None else 0
        logging.info('Lattice with %d states and %d arcs constructed in %.3f seconds' % (
            lattice_size,
            openfst.num_arcs(sentence_lattice),
            time.time() - start_time)
        )

        if lattice_size == 0:
            logging.error('Lattice is empty.')
            return None

        # Add start and end of sentence symbols before LM application
        sentence_lattice = openfst.add_start_and_end_of_sentence_symbols(sentence_lattice)

        return sentence_lattice

    def construct_cell(self, cell, sentence, feature_weights_dict, local_pruning_lm, openfst, hiero_cells=None,
                       reference_subs_fsa=None, established_intersections=None):
        """
        Create a coverage cell RTN.
        :param cell: Cell object
        :param sentence: Sentence object
        :param feature_weights_dict: Dictionary of feature names and their weights
        :param local_pruning_lm: Language model FSA for local pruning
        :param openfst: OpenFST utility functions object
        :param hiero_cells: Hiero space of coverage cells
        :param reference_subs_fsa: FSA encoding all reference translation substrings. Not used.
        :param established_intersections: Dictionary of established intersection points
        :return: Cell RTN object
        """

        logging.debug('Creating coverage cell fst from %d rules for coverage %s (%s).' % (
            len(cell.rules),
            cell.bit_coverage,
            cell.int_coverage
        ))

        # Create a cell RTN from rules with the same bit coverage
        self.create_cell_rtn(cell, feature_weights_dict, openfst)

        # Save RTN to disk if set
        if self.write_rtn is not None:
            helpers.store_rtn(cell.rtn, self.write_rtn, cell.int_coverage, sentence.id, openfst)

        # If enabled, attempt local pruning if current cell is not at the top of derivation tree
        if self.local_pruning and not cell.is_top_cell:

            localpruning.local_prune(
                cell,
                self.local_pruning_settings.get('conditions'),
                self.local_pruning_settings.get('n_best'),
                local_pruning_lm,
                feature_weights_dict['language_model_probability'],
                openfst,
                hiero_cells=hiero_cells,
                established_intersections=established_intersections,
                extended_integration=self.extended_integration_opt,
            )

        logging.debug('%d states and %d arcs in cell before reduction' % (
            openfst.num_states(cell.rtn),
            openfst.num_arcs(cell.rtn)
        ))

        # Remove epsilon transitions, minimize, and determinize the RTN
        cell.rtn = openfst.rmep_min_det(cell.rtn)

        num_nonterminal_arcs, num_nonterminal_arcs_distinct = openfst.count_nonterminal_arcs(cell.rtn)

        logging.debug('%d states and %d arcs in cell after reduction. %d nonterminal arcs (%d distinct).' % (
            openfst.num_states(cell.rtn),
            openfst.num_arcs(cell.rtn),
            num_nonterminal_arcs,
            num_nonterminal_arcs_distinct
        ))

        # for path in openfst.get_paths(cell.rtn, output=False)[:20]:
        #     logging.debug(path)

        return cell.rtn


def store_and_cleanup(input_dir, output_dir, sentence_id_map, gzip=False, tmp_dir=None):
    """
    Store lattices and clean temporary directories
    :param input_dir: Post-LM lattice temporary directory path
    :param output_dir: Permanent lattice storage directory path
    :param sentence_id_map: Dictionary mapping sentence ids to filename ids
    :param gzip: Whether to compress FSTs in the final directory
    :param tmp_dir: Temporary directory path to remove
    :return:
    """

    # Inverse the sentence id mapping and transform it into filenames
    mapping = {v + '.fst': k + '.fst' for k, v in sentence_id_map.items()}

    logging.debug('Renaming results of LM composition')

    # Move fsts to a permanent location and rename filenames to their respective sentence id
    utility.move_and_rename_files(input_dir, output_dir, mapping, gzip)

    # Remove temporary directory
    if tmp_dir is not None:
        utility.remove_directory(tmp_dir)
