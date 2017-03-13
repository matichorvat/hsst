import logging
import functools

from hsst.decoding import integration
from hsst.decoding.cell_selection import create_derivation_tree
from hsst.preprocessing.filtering import SourceGraphSizeFilter


class Decoder(object):

    def __init__(self, testing_mode, temporary_storage, local_pruning=False, local_pruning_settings=None,
                 top_level_pruning=None, filter_min=0, filter_max=0, timeout=0, write_rtn=None, hiero_peeping=False,
                 expanded_integration=False, heuristic_intersection=False, hiero_only=False, gzip=False,
                 max_cell_coverage=0, carg_rules=False, hiero_backoff=False):
        """
        Initialize the decoder.
        :param testing_mode: True if decoder is run in testing mode
        :param temporary_storage: Path to temporary directory for storing translation and post-LM lattices
        :param local_pruning: Whether to use local pruning to reduce search space
        :param local_pruning_settings: Dictionary of settings for local pruning
        :param top_level_pruning: Settings for top level pruning if to be used
        :param filter_min: Minimum source graph size. If zero, filter is not applied
        :param filter_max: Maximum source graph size. If zero, filter is not applied
        :param timeout: Maximum running time in seconds for decoding of a single sentence using HSST.
                        Default is 300 seconds.
        :param write_rtn: Path string where RTNs will be written to disk if set
        :param hiero_peeping: Whether to expand HSST space by looking at compatible Hiero space during construction
        :param expanded_integration: Whether to use more comprehensive integration of HSST and Hiero spaces
        :param heuristic_integration: Whether to use heuristic alignment during integration
        :param hiero_only: Only use Hiero RTNs to translate the sentence
        :param gzip: Whether to gzip the final output lattices
        :param max_cell_coverage: Maximum number of nodes covered by a cell. Default of 0 does not limit it.
        :param carg_rules: If True, CARG rules will be used for translation
        :param hiero_backoff: If True, decode with hiero RTNs (if available) in case HSST fails
        :return:
        """

        logging.info('Initializing decoder.')

        self.testing_mode = testing_mode
        self.temporary_storage = temporary_storage

        self.top_level_pruning = top_level_pruning

        self.local_pruning = local_pruning
        if local_pruning:
            self.local_pruning_settings = local_pruning_settings

        if filter_min > 0 or filter_max > 0:
            self.filtering = SourceGraphSizeFilter(min_nodes=filter_min, max_nodes=filter_max)
        else:
            self.filtering = None

        self.timeout = timeout if timeout > 0 else 300

        self.write_rtn = write_rtn

        self.hiero_peeping = hiero_peeping
        self.extended_integration_opt = expanded_integration
        self.heuristic_intersection = heuristic_intersection

        self.hiero_only = hiero_only

        self.gzip = gzip

        self.max_cell_coverage = max_cell_coverage if max_cell_coverage is not None else 0

        self.carg_rules = carg_rules
        self.hiero_backoff = hiero_backoff

    def construct_hsst_space(self, sentence, sentence_grammar, sentence_coverage, feature_weights_dict, local_pruning_lm,
                             openfst, hiero_cells=None, reference_subs_fsa=None, established_intersections=None):
        """
        Construct HSST search space by constructing individual RTN cells.
        :param sentence: Sentence object
        :param sentence_grammar: Dictionary of rule_id:rule instances for the sentence
        :param sentence_coverage: Dictionary of coverages with associated rules for the sentence
        :param feature_weights_dict: Dictionary of feature names and their weights
        :param local_pruning_lm: Language model FSA for local pruning
        :param openfst: OpenFST utility functions object
        :param hiero_cells: Hiero space of coverage cells
        :param reference_subs_fsa: FSA encoding all reference translation substrings
        :param established_intersections: Dictionary of established intersection points
        :return: DerivationTree object
        """
        # Load hiero peepholes if configured
        hiero_peepholes = None
        if hiero_cells is not None and self.hiero_peeping:
            logging.debug('Obtaining hiero peepholes.')
            hiero_peepholes = integration.hiero_peepholes(
                sentence,
                hiero_cells.keys(),
                heuristic_intersection=self.heuristic_intersection
            )

        # Create derivation tree
        derivation_tree = create_derivation_tree(
            sentence_grammar,
            sentence_coverage,
            hiero_peepholes=hiero_peepholes,
            max_cell_coverage=self.max_cell_coverage,
            allow_carg_rules=self.carg_rules
        )

        logging.debug('There are %d HSST cells to create after cell selection, with %s (%s) as top cell.' % (
            len(derivation_tree),
            derivation_tree.top_hsst_cell.bit_coverage if derivation_tree.top_hsst_cell is not None else 'None',
            derivation_tree.top_hsst_cell.int_coverage if derivation_tree.top_hsst_cell is not None else 'None'
        ))

        if hiero_cells:
            # Attach hiero RTN labels to HSST cells where the two spaces intersect
            integration.simple_hiero_into_hsst_integration(
                sentence,
                derivation_tree,
                hiero_cells,
                heuristic_intersection=False
            )

        # Create an FST for every bit coverage and store it in coverage_cells
        for int_coverage in derivation_tree.get_ordered_int_coverages():

            # Create coverage cell RTN
            self.construct_cell(
                derivation_tree[int_coverage],
                sentence,
                feature_weights_dict,
                local_pruning_lm,
                openfst,
                hiero_cells,
                reference_subs_fsa,
                established_intersections
            )

        return derivation_tree

    def construct_cell(self, cell, sentence, feature_weights_dict, local_pruning_lm, openfst,
                       hiero_space=None, reference_subs_fsa=None, established_intersections=None):
        """
        Create a coverage cell RTN.
        :param cell: Cell object
        :param sentence: Sentence object
        :param feature_weights_dict: Dictionary of feature names and their weights
        :param local_pruning_lm: Language model FSA for local pruning
        :param openfst: OpenFST utility functions object
        :param hsst_space: HSST space of coverage cells
        :param hiero_space: Hiero space of coverage cells
        :param reference_subs_fsa: FSA encoding all reference translation substrings. Not used.
        :param established_intersections: Dictionary of established intersection points
        :return: Cell RTN object
        """

        raise NotImplementedError()

    def create_cell_rtn(self, cell, feature_weights_dict, openfst):
        """
        Create cell RTN by unioning rule RTNs and store it in Cell object.
        :param cell: Cell object
        :param feature_weights_dict: Dictionary of feature names and their weights
        :param openfst: OpenFST utility functions object
        :param hsst_space: HSST space of coverage cells
        :param hiero_peepholes: Dictionary of HSST and Hiero space intersections
        """

        cell_rtn = None
        rule_count = 0

        for rule in cell.rules:
            # Create rule RTN FST
            rule_rtn = openfst.create_rule_fst(rule, feature_weights_dict)
            rule_count += 1

            # Union the rule RTN with cell RTN
            if cell_rtn is None:
                cell_rtn = rule_rtn
            else:
                openfst.union(cell_rtn, rule_rtn)

        logging.debug('%d rules added to the cell RTN.' % rule_count)

        # If there exist intersections of this cell with hiero, attach them via unioning
        if len(cell.intersections) > 0:
            for hiero_label in cell.intersections:
                integration.integrate_hiero_into_hsst_rtn(cell_rtn, hiero_label, openfst, feature_weights_dict)

        cell.rtn = cell_rtn
