import fst
import logging
import operator

from hsst.utility import utility


class OpenFST(object):

    PHI = 999999996
    OOV = 999999998
    DR = 999999999

    def __init__(self, wmap_filename, lm_vcb_filename):
        """
        Initialize OpenFST object
        :param wmap_filename: Path to the word map
        :param lm_vcb_filename: Path to the LM vocabulary
        :return:
        """

        if wmap_filename is not None:
            self.wmap = self.read_wmap(wmap_filename)
        else:
            self.wmap = None

        if lm_vcb_filename is not None:
            self.lm_vocab = set(
                int(x.decode('utf-8')) for x in open(lm_vcb_filename).read().split('\n') if x.strip() != '')
        else:
            self.lm_vocab = None

    def store_lattice(self, lattice, lattice_filename, gzip=False, overwrite=True):
        self.write_fst(lattice, lattice_filename)

        if gzip:
            utility.gzip_file(lattice_filename, overwrite)

    def get_paths(self, fst_1, output_labels=True, ignore_epsilons=True):
        paths = list()

        for i, path in enumerate(fst_1.paths()):
            path_string_idx = ' '.join(str(arc.olabel) if output_labels else str(arc.ilabel) for arc in path)

            tokens = []
            for arc in path:
                idx = arc.olabel if output_labels else arc.ilabel

                if ignore_epsilons and idx == 0:
                    continue

                token = self.wmap[idx] if self.wmap is not None and idx in self.wmap else str(idx)
                tokens.append(token)

            path_string = ' '.join(tokens)

            if len(path) > 0:
                path_weight = float(reduce(operator.mul, (arc.weight for arc in path)))
            else:
                path_weight = None

            paths.append((path_string_idx, path_string, path_weight))

        return paths

    def create_n_best_list(self, fst_1, n_best, output=False):
        """
        Compute an n-best list from an FST.
        :param fst_1: Input FST
        :param n_best: Number of best outputs
        :param output: If True, use output labels otherwise input labels
        :return: List of length n_best
        """

        n_best_fst = self.shortest_path(fst_1, n_best)
        n_best_fst = self.rmep_min_det(n_best_fst)
        self.top_sort(n_best_fst)

        # Get n-best paths and their weights, sort from lowest to highest weight
        n_best_list = sorted(self.get_paths(n_best_fst, output_labels=output), key=lambda x: x[2])

        return n_best_list

    @staticmethod
    def read_wmap(filename):
        """
        Read a word map from file
        :param filename: Path to the WMAP
        :return: WMAP dictionary
        """

        wmap = dict()

        with open(filename, 'rb') as stf:
            for line in stf:
                line = line.strip()

                if line != '':
                    try:
                        line = line.split('\t')
                        idx = int(line[0])
                        symbol = line[1].decode('utf-8')

                        wmap[idx] = symbol

                    except ValueError:
                        logging.warn('Could not cast integer: %s for symbol %s.' % (line[0], line[1]))

        return wmap

    @staticmethod
    def read_fst(filename, symbol_tables=False):
        fst_1 = fst.read_std(filename)

        if not symbol_tables:
            fst_1.isyms = None
            fst_1.osyms = None

        return fst_1

    @staticmethod
    def write_fst(fst_1, filename):
        fst_1.write(filename)

    @staticmethod
    def create_acceptor():
        return fst.Acceptor()

    def phi_compose(self, fst_1, fst_2):
        return fst_1.phi_compose(fst_2, self.PHI)

    def prune_with_shortest_path(self, fst_1, prune_threshold, n, unique=True):
        fst_shortest_paths = self.shortest_path(fst_1, n, unique)
        self.prune(fst_1, prune_threshold)
        self.union(fst_1, fst_shortest_paths)

    def rmep_min_det(self, fst_1, min_det=True):
        if fst_1 is None:
            return None

        fst_1.remove_epsilon()

        if min_det:
            fst_1 = self.transducer_det_min(fst_1)

        return fst_1

    @staticmethod
    def transducer_det_min(fst_1):
        return fst_1.determinize_minimize_transducer()

    # @staticmethod
    # def determinize(fst_1):
    #     det_fst = fst_1.determinize()
    #     return det_fst

    # @staticmethod
    # def minimize(fst_1):
    #     fst_1.minimize()

    @staticmethod
    def rmepsilon(fst_1):
        fst_1.remove_epsilon()

    @staticmethod
    def compose(fst_1, fst_2):
        return fst_1.compose(fst_2)

    @staticmethod
    def intersect(fst_1, fst_2):
        return fst_1.intersect(fst_2)

    @staticmethod
    def shortest_path(fst_1, n=1, unique=True):
        return fst_1.shortest_path(n, unique)

    @staticmethod
    def top_sort(fst_1):
        fst_1.top_sort()

    @staticmethod
    def union(fst_1, fst_2):
        fst_1.set_union(fst_2)

    @staticmethod
    def prune(fst_1, threshold):
        fst_1.prune(threshold)

    @staticmethod
    def replace(fst_1, mapping, epsilon=True):
        return fst_1.replace(mapping, epsilon=epsilon)

    @staticmethod
    def remove_weights(fst_1):
        return fst_1.remove_weights()

    @staticmethod
    def arc_sort(fst_1, output=True):
        if output:
            fst_1.arc_sort_output()
        else:
            fst_1.arc_sort_input()

        return fst_1

    @staticmethod
    def custom_arc_map(fst_1, map_func):
        fst_mapped = fst_1.copy()

        for state in fst_mapped.states:
            for arc in state.arcs:
                arc.weight = fst.TropicalWeight(map_func(float(arc.weight)))

        return fst_mapped

    @staticmethod
    def invert_weights(fst_1):
        return fst_1.invert_weights()

    @staticmethod
    def add_start_and_end_of_sentence_symbols(fst_1):
        """
        Concatenate start (beginning) and end (end) of sentence symbols to the FST.
        :param fst_1: FST object
        :return: FST with prepended start of sentence symbol and appended end of sentence symbol.
        """

        raise NotImplementedError()

    @staticmethod
    def project(fst_1, output=True):
        if output:
            fst_1.project_output()
        else:
            fst_1.project_input()

        return fst_1

    @staticmethod
    def num_states(fst_1):
        return len(fst_1)

    @staticmethod
    def num_arcs(fst_1):
        return fst_1.num_arcs()

    @staticmethod
    def relabel(fst_1, input_symbol_map, output_symbol_map):
        fst_1.relabel(input_symbol_map, output_symbol_map)


