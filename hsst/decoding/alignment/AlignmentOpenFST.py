from collections import Counter

import fst

from hsst.decoding.OpenFST import OpenFST
from hsst.decoding import helpers


class AlignmentOpenFST(OpenFST):

    def __init__(self, wmap_filename, lm_vcb_filename, rule_id_offset=0):
        """
        Initialize AlignmentOpenFST object
        :param wmap_filename: Path to the word map
        :param lm_vcb_filename: Path to the LM vocabulary
        :param rule_id_offset: Starting rule id for HSST rules
        :return:
        """
        super(AlignmentOpenFST, self).__init__(wmap_filename, lm_vcb_filename)
        self.rule_id_offset = rule_id_offset

    def fst_tostring(self, fst_1, idx=False):
        """
        Construct a string describing the FST.
        :param fst_1: Input FST object
        :param idx: Whether to not map labels using word map
        :return: String representation of the FST
        """

        fst_string = 'Transducer\n'
        for state in fst_1.states:
            for arc in state.arcs:
                olabel = self.wmap[arc.olabel].encode('utf-8') if not idx and arc.olabel in self.wmap else arc.olabel

                fst_string += '{} -> {} / {} : {} / {}\n'.format(state.stateid, arc.nextstate, arc.ilabel, olabel,
                                                                 float(arc.weight))

            if state.final:
                fst_string += '%s / %s\n' % (state.stateid, state.final)

        fst_string += '-------\n'

        return fst_string

    @staticmethod
    def create_empty_fst():
        empty_fst = fst.Transducer()
        empty_fst.add_arc(0, 1, 0, 0)
        empty_fst[1].final = True

        return empty_fst

    @staticmethod
    def create_root_fst(label, int_coverage_cells):
        """
        Create a root FST consisting of a single (nonterminal) transition
        :param label: Nonterminal transition label
        :param int_coverage_cells: Dictionary of integer coverages and associated FSTs
        :return: Root FST
        """

        root_fst = fst.Transducer(isyms=fst.SymbolTable(), osyms=fst.SymbolTable())
        root_fst.osyms[label] = int(label)

        # Adding epsilon input label using symbol table lookup for id=0
        root_fst.add_arc(0, 1, root_fst.isyms.find(0), label)
        root_fst[1].final = True

        # Create root FST symbol table
        for int_coverage, cell_fst in int_coverage_cells.items():
            root_fst.osyms[int_coverage] = int(int_coverage)

        return root_fst

    def create_rule_fst(self, rule, feature_weights_dict):
        """
        Create rule FST accepting the sequence of target side tokens.
        :param rule: Rule object
        :param feature_weights_dict: Dictionary of feature names and their weights
        :return: Rule FST
        """

        # Determine whether to use word insertion penalty
        if 'word_insertion_penalty' in feature_weights_dict and not rule.hiero_intersection_rule:
            wip = feature_weights_dict['word_insertion_penalty']
        else:
            wip = None

        # Offset rule_id by rule_id_offset to prevent clashes with Hiero rule id space
        rule_id = rule.id
        if not rule.hiero_intersection_rule:
            rule_id += self.rule_id_offset

        rule_fst = fst.Transducer()

        # Insert rule arc at the start of the transducer (rule_id:epsilon)
        rule_fst.add_arc(0, 1, int(rule_id), 0)

        # Add arcs representing target tokens one after the other
        # Index is adjusted to account for rule arc
        index = 1
        for index, token in enumerate(rule.target_side, 1):
            self.add_arc(rule_fst, index, token, rule.nonterminal_coverages, weight=wip)

        # Compute rule weight in a log linear model
        rule_weight = helpers.loglinear_rule_weight(rule.feature_dict, feature_weights_dict)

        # Add the rule weight to the final state in the FST
        rule_fst[index + 1].final = rule_weight

        if rule.hiero_intersection_rule:
            print rule_weight
            print self.fst_tostring(rule_fst)

        return rule_fst

    @staticmethod
    def add_arc(rule_fst, index, token, nonterminal_coverages, weight=None):
        """
        Add an arc to rule FST
        :param rule_fst: Rule FST being built
        :param index: Token index
        :param token: Token
        :param nonterminal_coverages: Dictionary of nonterminal symbols mapped to their bit coverages
        :param weight: Arc weight if specified (e.g. word insertion penalty)
        :return:
        """

        # Add arc of the form epsilon:token

        # Nonterminal symbol
        if token in nonterminal_coverages:
            rule_fst.add_arc(index, index + 1, 0, int(nonterminal_coverages[token]))

        elif int(token) == OpenFST.DR:
            rule_fst.add_arc(index, index + 1, 0, OpenFST.DR)

        elif weight is None:
            rule_fst.add_arc(index, index + 1, 0, int(token))

        # Terminal symbol
        else:
            rule_fst.add_arc(index, index + 1, 0, int(token), weight=-weight)

    @staticmethod
    def add_start_and_end_of_sentence_symbols(fst_1):
        """
        Concatenate start (beginning) and end (end) of sentence symbols to the FST.
        :param fst_1: FST object
        :return: FST with prepended start of sentence symbol and appended end of sentence symbol.
        """

        # Create start of sentence FSA
        # 1 is start of sentence label
        start_of_sentence = fst.Transducer()
        start_of_sentence.add_arc(0, 1, 0, 1)
        start_of_sentence[1].final = True

        # Create end of sentence FSA
        # 2 is end of sentence label
        end_of_sentence = fst.Transducer()
        end_of_sentence.add_arc(0, 1, 0, 2)
        end_of_sentence[1].final = True

        # Modify start_of_sentence by concatenating fst_1
        start_of_sentence.concatenate(fst_1)

        # Modify joint start_of_sentence and fst_1 by concatenating end_of_sentence
        start_of_sentence.concatenate(end_of_sentence)

        return start_of_sentence

    @staticmethod
    def count_nonterminal_arcs(fst_1):
        nt_arcs = Counter()
        for state in fst_1.states:
            for arc in state.arcs:
                label = str(arc.olabel)
                if len(label) >= 10:
                    nt_arcs[label] += 1

        return sum(nt_arcs.values()), len(nt_arcs)
