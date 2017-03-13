import time
import logging

from hsst.decoding import helpers
from hsst.decoding import integration


def local_prune(cell, local_pruning_conditions, n_best, local_pruning_lm, lm_feature_weight, openfst,
                hiero_cells=None, established_intersections=None, extended_integration=False):
    """
    Perform local pruning of cell RTN if any of the conditions for pruning apply.
    :param cell: Cell object to prune
    :param local_pruning_conditions: List of pruning condition (symbol, coverage_n, size_n, threshold_n) tuples
    :param n_best: Number of shortest paths to keep in every pruned cell regardless of pruning threshold
    :param local_pruning_lm: Language model FSA for local pruning
    :param lm_feature_weight: Language model feature weight
    :param openfst: OpenFST utility functions object
    :param hiero_cells: Hiero coverage cells
    :param established_intersections: Dictionary of established intersection points
    :param extended_integration: Whether to perform extended integration before local pruning
    """

    if local_pruning_conditions is None:
        logging.debug('No pruning conditions set, local pruning not performed.')
        return cell.rtn, False

    num_states = openfst.num_states(cell.rtn)
    bit_coverage_sum = sum([int(x) for x in list(cell.bit_coverage)])

    # Pruning conditions are ordered in reverse order of coverage size
    # Pruning is performed using the closest applicable condition
    for condition in local_pruning_conditions:
        symbol, coverage_n, size_n, threshold_n = condition

        # Pruning is performed using the closest applicable condition, symbol is ignored
        if bit_coverage_sum >= coverage_n:

            if num_states < size_n:
                break

            logging.debug('Pruning condition applied: %d >= %d and %d >= %d' % (bit_coverage_sum, coverage_n,
                                                                                num_states, size_n))

            current_time = time.time()

            # Perform pruning to obtain FST
            logging.debug('Starting pruning of cell with coverage %s' % cell.int_coverage)

            mapping = dict((label, cell.rtn) for label, cell in cell.subtree(unexpanded_only=True).items())

            if extended_integration:
                logging.debug('Starting extended integration')

                # Compute space intersections below the current cell but not including the current cell
                space_intersections = dict()
                for int_label, subtree_cell in cell.subtree(unexpanded_only=True).items():
                    if int_label == cell.int_coverage or len(subtree_cell.intersections) == 0:
                        continue

                    space_intersections[int_label] = subtree_cell.intersections

                mapping, _ = integration.extended_integration(
                    space_intersections,
                    hiero_cells,
                    mapping,
                    None,
                    openfst,
                    established_intersections
                )

            elif hiero_cells is not None:
                mapping.update(hiero_cells)

            carg_oov_tokens = {carg_token for carg_token in cell.subtree_carg_tokens() if carg_token not in openfst.lm_vocab}

            fsa = prune_rtn_with_lm(
                cell.int_coverage,
                mapping,
                local_pruning_lm,
                lm_feature_weight,
                n_best,
                threshold_n,
                openfst,
                carg_oov_tokens
            )

            if openfst.num_states(fsa) == 0:
                logging.error('FSA size after pruning is 0, likely due to using a symbol that is not in LM.')
                raise Exception('FSA size after pruning is 0, likely due to using a symbol that is not in LM.')

            else:
                cell.rtn = fsa
                cell.expanded_rtn = True

            logging.debug('Pruning finished in %.3f seconds' % (time.time() - current_time))

            break


def prune_rtn_with_lm(cell_label, mapping, lm_fsa, lm_feature_weight, n_best, pruning_threshold, openfst, carg_oov_tokens):
    """
    Prune cell RTN after expanding it to FSA and combining it with a LM.
    :param cell_label: Label of RTN to prune
    :param mapping: Dictionary of labels and associated FSTs
    :param lm_fsa: Language model FSA for local pruning
    :param lm_feature_weight: Language model feature weight
    :param n_best: Number of shortest paths to keep in pruned FST
    :param pruning_threshold: Pruning threshold value
    :param openfst: OpenFST utility functions object
    :param carg_oov_tokens: Set of integer CARG tokens that do not appear in language model
    :return: Pruned cell FSA
    """

    logging.info('Starting pruning of cell RTN with LM.')

    # Expand RTN into FSA
    logging.debug('Creating cell FST')
    fst = helpers.create_lattice(mapping, cell_label, openfst)

    # Relabel output DR symbols to epsilon so that it can be composed with LM FSA
    relabel_map = {openfst.DR: 0}

    # Relabel CARG tokens that do not appear in LM vocab to OOV symbols so that they can be composed with LM FSA
    if carg_oov_tokens:
        carg_oov_relabel = dict((token, openfst.OOV) for token in carg_oov_tokens)
        relabel_map.update(carg_oov_relabel)
        logging.warn('OOV CARG symbols encountered in local pruning: %s.' % ','.join(str(x) for x in carg_oov_tokens))

    openfst.relabel(fst, {}, relabel_map)

    # Create cell specific LM FSA and cell specific LM FSA with negative weights
    logging.debug('Creating cell-specific LM FSAs')
    cell_lm_fsa, cell_lm_fsa_neg = create_cell_lm_fsa(lm_fsa, fst, lm_feature_weight, openfst)

    # Compose cell FST with LM FSA to get combined weight FSA
    logging.debug('Composing cell FST (%d states, %d arcs) with LM FSA (%d states, %d arcs).' % (
        openfst.num_states(fst),
        openfst.num_arcs(fst),
        openfst.num_states(cell_lm_fsa),
        openfst.num_arcs(cell_lm_fsa)
    ))

    fst_combined = openfst.compose(fst, cell_lm_fsa)

    # Project input labels to restore DR symbols on output as the following det and prune operations require FSA
    fst_combined = openfst.project(fst_combined, output=False)

    # Rmeps, det, min - necessary so that shortest path during pruning grabs truly unique paths
    fst_combined = openfst.rmep_min_det(fst_combined)

    # Prune the combined WFST in-place
    logging.debug('Pruning combined FST')
    fst_combined = helpers.fst_pruning(fst_combined, pruning_threshold, n_best, openfst)

    # Relabel output DR symbols and CARGs to OOVs to epsilon so that it can be composed with LM FSA
    openfst.relabel(fst_combined, {}, relabel_map)

    # Compose pruned FSA with negative weight LM FSA to remove LM weights
    logging.debug('Removing LM weights from pruned FST via composition')
    fst_final = openfst.compose(fst_combined, cell_lm_fsa_neg)

    # Project input labels to restore DR symbols on output
    fst_final = openfst.project(fst_final, output=False)

    # for path in openfst.get_paths(fst_final, output_labels=False)[:10]:
    #     logging.debug(path[1])

    return fst_final


def create_cell_lm_fsa(lm_fsa, cell_fst, lm_feature_weight, openfst):
    """
    Create cell specific LM FSA.
    :param lm_fsa: Full Language model FSA
    :param cell_fst: Cell FST
    :param lm_feature_weight: Language Model feature weight
    :param openfst: OpenFST utility functions object
    :return: Cell specific LM FSA
    """

    logging.info('Starting LM cell FSA creation.')

    # Remove weights from FST
    cell_fst_unw = openfst.remove_weights(cell_fst)

    # Compose unweighted cell FST with pruning LM FSA using PHI composition
    # Projecting on output creates the minimal needed LM FSA
    cell_lm_fsa = openfst.phi_compose(cell_fst_unw, lm_fsa)
    cell_lm_fsa = openfst.project(cell_lm_fsa, output=True)
    cell_lm_fsa = openfst.rmep_min_det(cell_lm_fsa)

    # If LM weight is not 1, scale the arc weights according to the LM weight
    if lm_feature_weight != 1.0:
        cell_lm_fsa = openfst.custom_arc_map(cell_lm_fsa, lambda x: x * lm_feature_weight)

    # Sort arcs for future composition
    cell_lm_fsa = openfst.arc_sort(cell_lm_fsa, output=False)

    # Create a negative weight LM FSA
    # This will be used for subtracting the LM weights from the pruned combined FSA
    cell_lm_fsa_neg = openfst.invert_weights(cell_lm_fsa)

    logging.debug('LM Cell FSA for pruning created.')

    return cell_lm_fsa, cell_lm_fsa_neg


# AlignmentDecoder pruning
# def prune_rtn_with_lm(self, rtn, bit_coverage, coverage_cells, lm_fsa, lm_feature_weight, pruning_threshold,
#                       openfst):
#     """
#     Prune cell RTN after expanding it to FSA and combining it with a LM.
#     :param rtn: Cell RTN FSA
#     :param bit_coverage: RTN bit coverage
#     :param coverage_cells: Dictionary of coverages and associated FSTs
#     :param lm_fsa: Language model FSA for local pruning
#     :param lm_feature_weight: Language model feature weight
#     :param pruning_threshold: Pruning threshold value
#     :param openfst: OpenFST utility functions object
#     :return: Pruned cell FSA
#     """
#
#     # Expand RTN into FSA
#     fsa = helpers.create_fsa(rtn, bit_coverage, coverage_cells, openfst)
#
#     # Create cell specific LM FSA and cell specific LM FSA with negative weights
#     cell_lm_fsa, cell_lm_fsa_neg = helpers.create_cell_lm_fsa(lm_fsa, fsa, lm_feature_weight, openfst)
#
#     # Intersect LM FSA with cell FSA to get combined weight FSA
#     fsa_combined = openfst.compose(fsa, cell_lm_fsa)
#     fsa_combined = openfst.rmep_min_det(fsa_combined)
#     logging.debug('Combined weight FSA for pruning created.')
#
#     # Prune the combined WFST in-place and reduce it
#     openfst.prune(fsa_combined, pruning_threshold)
#     fsa_combined = openfst.rmep_min_det(fsa_combined)
#     logging.debug('Combined weight FSA pruned.')
#
#     # Intersect negative weight LM FSA with pruned FSA to remove LM weights
#     fsa_final = openfst.compose(fsa_combined, cell_lm_fsa_neg)
#     fsa_final = openfst.rmep_min_det(fsa_final)
#     logging.debug('Final pruned FSA contains %d states.' % sum(1 for _ in fsa_final.states))
#
#     return fsa_final