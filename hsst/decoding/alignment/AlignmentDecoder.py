import os
import time
import logging

from hsst.decoding import helpers
from hsst.decoding import integration
from hsst.decoding.Decoder import Decoder
from hsst.utility import utility
from hsst.utility import timeout as to

ALILATS_STORAGE_DIR = 'ALILATS/'


class MissingFileException(Exception):
    pass


class AlignmentDecoder(Decoder):

    def initialize_tmp_dirs(self, subdir):
        """
        Creates temporary directories for storing translation and post-LM lattices.
        :param subdir: If specified, creates temporary directories inside a subdirectory.
                        Primarily used for using the same decoder multiple times on a single set
        :return: Paths to the translation and post-LM lattices temporary directories
        """

        if subdir is None:
            lats_dir = os.path.join(self.temporary_storage, ALILATS_STORAGE_DIR)
        else:
            lats_dir = os.path.join(self.temporary_storage, subdir, ALILATS_STORAGE_DIR)

        utility.create_dir(lats_dir)

        logging.debug('Temporary directories %s.' % lats_dir)

        return lats_dir

    def __call__(self, dataset, grammar_path, coverage_path, feature_weights_dict, openfst, translation_lats,
                       hiero_dir=None, hiero_lats=None, subdir=None, prune_reference_shortest_path=1000, outdir=None):
        """
        Decode a dataset in alignment mode.
        :param dataset: Dataset object containing sentences to decode
        :param grammar_path: Path to the sentence specific grammars
        :param coverage_path: Path to the sentence specific coverages
        :param feature_weights_dict: Dictionary of feature names and their weights
        :param openfst: OpenFST utility functions object
        :param translation_lats: Directory path storing translation lattices
        :param hiero_dir: Hiero RTN directory path
        :param hiero_lats: Directory path storing Hiero translation lattices - used for hiero only mode
        :param subdir: If specified, creates temporary directories inside a subdirectory
        :param prune_reference_shortest_path: Number of best reference hypotheses extracted from translation FSA
        :return:
        """

        logging.info('Starting decoding of the dataset in alignment mode.')

        # Create temporary directories
        alilats_dir = self.initialize_tmp_dirs(subdir)

        # Iterate over the sentences and attempt to decode them
        for index, sentence in enumerate(dataset):
            self.decode_sentence(
                index,
                sentence,
                grammar_path,
                coverage_path,
                feature_weights_dict,
                None,
                openfst,
                translation_lats,
                alilats_dir,
                hiero_dir,
                hiero_lats,
                prune_reference_shortest_path
            )

        # Store final lattices if required
        store_and_cleanup(alilats_dir, outdir, self.gzip)

    def decode_sentence(self, index, sentence, grammar_path, coverage_path, feature_weights_dict, local_pruning_lm,
                        openfst, translation_lats, alilats_dir, hiero_dir, hiero_lats, prune_reference_shortest_path):
        """
        Decode a single sentence in alignment mode and save its lattice to disk.
        :param index: Sentence index in the dataset
        :param sentence: Sentence object
        :param grammar_path: Path to the sentence specific grammars
        :param coverage_path: Path to the sentence specific coverages
        :param feature_weights_dict: Dictionary of feature names and their weights
        :param local_pruning_lm: Language model FSA for local pruning
        :param openfst: OpenFST utility functions object
        :param translation_lats: Directory path storing translation lattices
        :param alilats_dir: Directory path for storing alignment lattices
        :param hiero_lats: Directory path storing Hiero translation lattices - used for hiero only mode
        :param hiero_dir: Hiero RTN directory path
        :param prune_reference_shortest_path: Number of best reference hypotheses extracted from translation FSA
        :return:
        """

        logging.info('Starting decoding of sentence with index %d and id %s' % (index, sentence.id))
        start_time = time.time()

        if self.hiero_only:
            logging.info('Starting decoding of sentence %s in hiero only mode.' % (sentence.id,))
            sentence_lattice = self.hiero_only_mode(
                sentence,
                hiero_dir,
                hiero_lats,
                prune_reference_shortest_path,
                openfst
            )

        elif self.filtering is not None and not self.filtering.filter(sentence):
            # If sentence is filtered out for HSST decoding, try to use hiero
            logging.info('Sentence %s filtered out of HSST decoding.' % (sentence.id,))

            if hiero_dir is None:
                logging.info('Returning empty result for sentence %s.' % (sentence.id,))
                return

            else:
                logging.info('Starting decoding of sentence %s in hiero only mode.' % (sentence.id,))
                sentence_lattice = self.hiero_only_mode(
                    sentence,
                    hiero_dir,
                    hiero_lats,
                    prune_reference_shortest_path,
                    openfst
                )

        else:

            hiero_subdir = None
            if hiero_dir is not None:
                hiero_subdir = os.path.join(hiero_dir, sentence.id)

            try:
                # Load sentence specific grammar
                logging.info('Loading grammar for sentence %s from %s.' % (sentence.id, grammar_path))
                sentence_grammar = utility.load_sentence_specific_grammar(sentence.id, grammar_path)
                logging.info('Loading coverage for sentence %s from %s.' % (sentence.id, coverage_path))
                sentence_coverage = utility.load_sentence_specific_coverage(sentence.id, coverage_path)

                with to.timeout(seconds=self.timeout):
                    # Create reference and reference substring FSAs
                    reference_fsa, reference_subs_fsa = self.create_reference_acceptors(
                        sentence.id,
                        translation_lats,
                        openfst,
                        prune_reference_shortest_path
                    )

                    if hiero_subdir is None:
                        logging.info('Starting decoding of sentence %s in HSST only mode.' % (sentence.id,))
                        sentence_lattice = self.hsst_only_mode(
                            sentence,
                            sentence_grammar,
                            sentence_coverage,
                            feature_weights_dict,
                            local_pruning_lm,
                            reference_fsa,
                            reference_subs_fsa,
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
                            reference_fsa,
                            reference_subs_fsa,
                            openfst,
                            hiero_subdir
                        )

            except to.TimeoutError:
                logging.warn('Decoding of sentence %s (%s) was terminated due to exceeding the timeout limit of %d seconds' % (
                    sentence.id, sentence.orig_id, self.timeout))

                if hiero_dir is None or not self.hiero_backoff:
                    sentence_lattice = openfst.create_empty_fst()

                else:
                    # Relaunch decoding using Hiero-only mode
                    logging.info('Starting decoding of sentence %s in hiero only mode.' % (sentence.id,))
                    sentence_lattice = self.hiero_only_mode(
                        sentence,
                        hiero_dir,
                        hiero_lats,
                        prune_reference_shortest_path,
                        openfst
                    )

            except MemoryError:
                logging.error('Decoding of sentence %s (%s) was terminated due to exceeding allowed memory after %d seconds' % (
                    sentence.id, sentence.orig_id, time.time() - start_time))

                if hiero_dir is None or not self.hiero_backoff:
                    sentence_lattice = openfst.create_empty_fst()

                else:
                    # Relaunch decoding using Hiero-only mode
                    logging.info('Starting decoding of sentence %s in hiero only mode.' % (sentence.id,))
                    sentence_lattice = self.hiero_only_mode(
                        sentence,
                        hiero_dir,
                        hiero_lats,
                        prune_reference_shortest_path,
                        openfst
                    )

            except MissingFileException:
                logging.exception('Missing file')

                if hiero_dir is None or not self.hiero_backoff:
                    sentence_lattice = openfst.create_empty_fst()

                else:
                    # Relaunch decoding using Hiero-only mode
                    logging.info('Starting decoding of sentence %s in hiero only mode.' % (sentence.id,))
                    sentence_lattice = self.hiero_only_mode(
                        sentence,
                        hiero_dir,
                        hiero_lats,
                        prune_reference_shortest_path,
                        openfst
                    )

        # If decoding failed, skip storing lattice to the disk
        if sentence_lattice is None:
            logging.error('Decoding of sentence with index %d and id %s failed in %.3f seconds' % (
                index, sentence.id, time.time() - start_time))
            sentence_lattice = openfst.create_empty_fst()

        logging.info('Decoding of sentence with index %d and id %s finished in %.3f seconds' % (
            index, sentence.id, time.time() - start_time))

        # Store lattice to disk
        current_time = time.time()
        lattice_path = os.path.join(alilats_dir, '%s.fst' % sentence.id)
        openfst.store_lattice(sentence_lattice, lattice_path)

        logging.info('Lattice stored to disk as %s in %.3f seconds' % (lattice_path, time.time() - current_time))

    def hiero_only_mode(self, sentence, hiero_dir, hiero_lats, prune_reference_shortest_path, openfst):
        """
        Create an alignment transducer for the given sentence using only Hiero RTNs.
        :param sentence: Sentence object
        :param hiero_dir: Hiero RTN directory path
        :param hiero_lats: Directory path storing Hiero translation lattices
        :param prune_reference_shortest_path: Number of best reference hypotheses extracted from translation FSA
        :param openfst: OpenFST utility functions object
        :return:o
        """

        start_time = time.time()

        # Create reference and reference substring FSAs
        reference_fsa, _ = self.create_reference_acceptors(
            sentence.id,
            hiero_lats,
            openfst,
            prune_reference_shortest_path
        )

        # Load hiero RTNs
        hiero_subdir = os.path.join(hiero_dir, sentence.id)
        hiero_cells, top_hiero_cell_label = integration.construct_hiero_space(hiero_subdir, openfst)

        logging.debug('Constructing a lattice out of %d RTNs, top cell is %s.' %
                      (len(hiero_cells), top_hiero_cell_label))

        # Construct Hiero FST
        sentence_lattice = helpers.create_lattice(hiero_cells, top_hiero_cell_label, openfst)

        # Compose lattice with full sentence acceptor to only allow derivations of previously translated sentences
        logging.info('Composing lattice with reference FSA.')
        sentence_lattice = openfst.compose(sentence_lattice, reference_fsa)
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
                       reference_fsa, reference_subs_fsa, openfst):
        """
        Create a translation transducer (lattice) mapping rule ids to target strings for the given sentence.
        :param sentence: Sentence object
        :param sentence_grammar: Dictionary of rule_id:rule instances for the sentence
        :param sentence_coverage: Dictionary of coverages with associated rules for the sentence
        :param feature_weights_dict: Dictionary of feature names and their weights
        :param local_pruning_lm: Language model FSA for local pruning
        :param reference_fsa: FSA encoding reference translation strings
        :param reference_subs_fsa: FSA encoding all reference translation substrings
        :param openfst: OpenFST utility functions object
        :return: Sentence FST object
        """

        start_time = time.time()

        # Populate the coverage cells with HSST RTNs
        derivation_tree = self.construct_hsst_space(
            sentence,
            sentence_grammar,
            sentence_coverage,
            feature_weights_dict,
            local_pruning_lm,
            openfst,
            reference_subs_fsa=reference_subs_fsa
        )

        # Lattice has already been created from RTNs because of substring acceptor composition
        # Add start and end of sentence symbols before composing with complete reference string FSA
        sentence_lattice = openfst.add_start_and_end_of_sentence_symbols(derivation_tree.top_hsst_cell.rtn)

        # Compose lattice with full sentence acceptor to only allow derivations of previously translated sentences
        logging.info('Composing lattice with reference FSA.')
        sentence_lattice = openfst.compose(sentence_lattice, reference_fsa)
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

    def hsst_hiero_mode(self, sentence, sentence_grammar, sentence_coverage, feature_weights_dict, local_pruning_lm,
                        reference_fsa, reference_subs_fsa, openfst, hiero_subdir=None):
        """
        Create a translation transducer (lattice) mapping rule ids to target strings for the given sentence.
        :param sentence: Sentence object
        :param sentence_grammar: Dictionary of rule_id:rule instances for the sentence
        :param sentence_coverage: Dictionary of coverages with associated rules for the sentence
        :param feature_weights_dict: Dictionary of feature names and their weights
        :param local_pruning_lm: Language model FSA for local pruning
        :param reference_fsa: FSA encoding reference translation strings
        :param reference_subs_fsa: FSA encoding all reference translation substrings
        :param openfst: OpenFST utility functions object
        :param hiero_subdir: Sentence Hiero RTN directory path
        :return: Sentence FSA (lattice) object
        """

        start_time = time.time()

        # Load hiero RTNs
        logging.debug('Constructing hiero space.')
        hiero_cells, top_hiero_cell_label = integration.construct_hiero_space(hiero_subdir, openfst)

        # Track added hiero->hsst links if using local pruning and extended integration to prevent duplicating them
        if self.extended_integration_opt:
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
            reference_subs_fsa=reference_subs_fsa,
            established_intersections=established_intersections
        )

        # Lattice has already been created from RTNs because of substring acceptor composition
        # Add start and end of sentence symbols before composing with complete reference string FSA
        sentence_lattice = openfst.add_start_and_end_of_sentence_symbols(derivation_tree.top_hsst_cell.rtn)

        # Compose lattice with full sentence acceptor to only allow derivations of previously translated sentences
        logging.info('Composing lattice with reference FSA.')
        sentence_lattice = openfst.compose(sentence_lattice, reference_fsa)
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

        mapping = dict((label, cell.rtn) for label, cell in cell.subtree(unexpanded_only=True).items())

        if hiero_cells is not None:
            mapping.update(hiero_cells)

        if self.extended_integration_opt:
            logging.debug('Starting extended integration')

            # Compute space intersections below the current cell but not including the current cell
            space_intersections = dict()
            for int_label, subtree_cell in cell.subtree(unexpanded_only=True).items():
                if int_label == cell.int_coverage or len(subtree_cell.intersections) == 0:
                    continue

                space_intersections[int_label] = subtree_cell.intersections

            if len(space_intersections) > 0:
                mapping, _ = integration.extended_integration(
                    space_intersections,
                    hiero_cells,
                    mapping,
                    None,
                    openfst,
                    established_intersections
                )

        # Save RTN to disk if set
        if self.write_rtn is not None:
            helpers.store_rtn(cell.rtn, self.write_rtn, cell.int_coverage, sentence.id, openfst)

        num_nonterminal_arcs, num_nonterminal_arcs_distinct = openfst.count_nonterminal_arcs(cell.rtn)

        logging.debug('%d states and %d arcs in cell before expansion. %d nonterminal arcs (%d distinct).' % (
            openfst.num_states(cell.rtn),
            openfst.num_arcs(cell.rtn),
            num_nonterminal_arcs,
            num_nonterminal_arcs_distinct
        ))

        # Expand RTN into FST
        logging.debug('Expanding RTN into FST.')
        cell_fst = helpers.create_lattice(mapping, cell.int_coverage, openfst)

        logging.debug('Composing cell FST (%d states, %d arcs) with reference substring FSA.' % (
            openfst.num_states(cell_fst),
            openfst.num_arcs(cell_fst)
        ))

        # Compose cell FST with reference substring FSA
        cell.rtn = openfst.compose(cell_fst, reference_subs_fsa)
        cell.expanded_rtn = True

        logging.debug('Cell FST with %d states and %d arcs created.' % (
            openfst.num_states(cell.rtn),
            openfst.num_arcs(cell.rtn)
        ))

        # Remove epsilon transitions, minimize, and determinize the RTN
        cell.rtn = openfst.rmep_min_det(cell.rtn)

        logging.debug('%d states and %d arcs in cell after reduction.' % (
            openfst.num_states(cell.rtn),
            openfst.num_arcs(cell.rtn)
        ))

    def create_reference_acceptors(self, sentence_id, translation_lats, openfst, prune_reference_shortest_path):
        """
        Create FSA that will accept complete reference strings and FSA that will accept all its substrings.
        :param sentence_id: Sentece ID string
        :param translation_lats: Directory path storing translation lattices
        :param openfst: OpenFST utility functions object
        :param prune_reference_shortest_path: Number of best reference hypotheses extracted from translation FSA
        :return: Tuple (reference FSA object, reference substring FSA object)
        """

        # Read sentence translation FSA from disk
        fsa_filename = translation_lats.replace('?', sentence_id)

        if not os.path.isfile(fsa_filename):
            raise MissingFileException('Lattice reference file %s does not exist.' % (fsa_filename,))

        if fsa_filename.endswith('.gz'):
            logging.info('Expanding sentence translation to disk.')
            expanded_fsa_filename = os.path.join(self.temporary_storage, sentence_id + '.fst')
            utility.ungzip_file(fsa_filename, expanded_fsa_filename)
            fsa_filename = expanded_fsa_filename

        logging.info('Reading sentence translation FSA from disk.')
        fsa = openfst.read_fst(fsa_filename)

        logging.debug('Sentence translation FSA has %d states and %d arcs.' % (
            openfst.num_states(fsa),
            openfst.num_arcs(fsa))
        )

        # Prune FSA by taking shortest paths
        if prune_reference_shortest_path:
            logging.info('Pruning sentence translation FSA, taking %d shortest paths.' % prune_reference_shortest_path)
            reference_fsa = openfst.shortest_path(fsa, prune_reference_shortest_path)
        else:
            reference_fsa = fsa

        # Remove weights from FSA
        logging.debug('Removing weights')
        reference_fsa = openfst.remove_weights(reference_fsa)

        # Create substring acceptor
        logging.info('Creating reference substring FSA.')
        reference_subs_fsa = self.create_reference_substring_acceptor(reference_fsa, openfst)

        # Remove epsilons to speed up later composition and sort by input labels
        reference_fsa = openfst.rmep_min_det(reference_fsa)
        reference_fsa = openfst.arc_sort(reference_fsa, output=False)

        logging.debug('Reference FSA has %d states and %d arcs.' % (
            openfst.num_states(reference_fsa),
            openfst.num_arcs(reference_fsa))
        )

        logging.debug('Reference substring FSA has %d states and %d arcs.' % (
            openfst.num_states(reference_subs_fsa),
            openfst.num_arcs(reference_subs_fsa))
        )

        return reference_fsa, reference_subs_fsa

    def create_reference_substring_acceptor(self, reference_fsa, openfst):
        """
        Create FSA that will accept all substrings of the complete reference FSA.
        :param reference_fsa: Reference FSA accepting complete reference strings
        :param openfst: OpenFST utility functions object
        :return: Reference substring FSA
        """

        # Create a copy of reference_fsa
        reference_subs_fsa = reference_fsa.copy()

        # Collect non-initial states
        non_initial_state_ids = set()

        for state in reference_subs_fsa.states:
            non_initial_state_ids.update(arc.nextstate for arc in state.arcs)

        # Change all non-initial states into final states
        for state_id in non_initial_state_ids:
            reference_subs_fsa[state_id].final = True

        # Create epsilon arcs from the initial state to all other states
        # The initial state id is 0

        for state in reference_subs_fsa.states:
            reference_subs_fsa.add_arc(0, state.stateid, 0, 0)

        # Remove epsilons to speed up later composition
        reference_subs_fsa = openfst.rmep_min_det(reference_subs_fsa)

        # Sort by input labels
        reference_subs_fsa = openfst.arc_sort(reference_subs_fsa, output=False)

        return reference_subs_fsa


def store_and_cleanup(alilats_dir, outdir, gzip=False):
    """
    Store lattices and clean temporary directories
    :param alilats_dir: Alignment lattice temporary directory path
    :param outdir: Permanent lattice storage directory path
    :param gzip: Whether to compress FSTs in the final directory
    :return:
    """

    logging.debug('Moving results from tmp dir %s to %s' % (alilats_dir, outdir))

    # Move fsts to a permanent location
    utility.move_directory(alilats_dir, outdir, gzip)
