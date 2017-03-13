import logging
from collections import defaultdict

from hsst.decoding import helpers


class DerivationTree(object):

    def __init__(self, hiero_peepholes=None, max_cell_coverage=0):
        self.hsst_cells = dict()
        self.top_hsst_cell = None
        self.hiero_peepholes = hiero_peepholes
        self.max_cell_coverage = max_cell_coverage
        self.space_intersections = None

        self.highest_bit_coverage_sum = 0

    def __getitem__(self, item):
        return self.hsst_cells[item]

    def __contains__(self, item):
        return item in self.hsst_cells

    def __len__(self):
        return len(self.hsst_cells)

    def get_ordered_int_coverages(self):
        # Order by increasing coverage length and starting from begining of the graph (e.g. 1000 before 0001)
        return sorted(self.hsst_cells.keys(), key=lambda x: (self.hsst_cells[x].coverage_len,
                                                             self.hsst_cells[x].bit_coverage[::-1]))

    def add_cell(self, int_coverage, rules):
        """
        Create a coverage cell and add rules to it. The assumption is that cells are created in order of increasing coverage.
        :param int_coverage: Integer coverage string
        :param rules: List of Rule objects
        """

        cell = Cell(int_coverage)

        # Filter out cells that cover more than self.max_cell_coverage nodes
        if self.max_cell_coverage > 0:
            bit_coverage_sum = sum([int(x) for x in list(cell.bit_coverage)])

            if bit_coverage_sum > self.max_cell_coverage:
                return

        # Add rules to the cell
        for rule in rules:
            if self.check_rule_nonterminals(rule):
                cell.add_rule(rule, self.hsst_cells)

        if len(cell.rules) > 0:
            # Add cell to derivation tree
            self.hsst_cells[cell.int_coverage] = cell

            # Update top cell
            self.update_top_cell(cell)

    def check_rule_nonterminals(self, rule):
        """
        Check that rule's nonterminal symbols point to existing cells in self.hsst_cells. If hiero peepholing is enabled,
        hiero cells count as well. In that case, rule's nonterminal coverage is updated to point to the hiero cell.
        :param rule: Rule object
        :return: Boolean
        """

        for nonterminal, nonterminal_coverage in rule.nonterminal_coverages.items():
            if nonterminal_coverage in self.hsst_cells:
                continue
            elif self.hiero_peepholes is not None and nonterminal_coverage in self.hiero_peepholes:
                rule.nonterminal_coverages[nonterminal] = self.hiero_peepholes[nonterminal_coverage]
                logging.debug('An additional rule used due to hiero peephole (%s).' % (nonterminal_coverage, ))
                continue
            else:
                return False

        return True

    def update_top_cell(self, cell):
        """
        Update derivation tree's top cell
        :param cell: New cell object
        """

        # Compute bit coverage sum
        bit_coverage_sum = sum([int(x) for x in list(cell.bit_coverage)])

        # Update highest_bit_coverage if necessary
        if bit_coverage_sum > self.highest_bit_coverage_sum:
            self.highest_bit_coverage_sum = bit_coverage_sum
            self.top_hsst_cell = cell


class Cell(object):

    def __init__(self, int_coverage, rtn=None, expanded_rtn=False):
        self.int_coverage = int_coverage
        self.bit_coverage = helpers.int_to_bit_coverage(int_coverage)
        self.rtn = rtn
        self.expanded_rtn = expanded_rtn
        self.carg_tokens = set()

        self.rules = []
        self.dependent_cells = {}
        self.intersections = []
        self.is_top_cell = False

    def __unicode__(self):
        return self.int_coverage + '|' + self.bit_coverage

    def __str__(self):
        return str(self.__unicode__())

    @property
    def coverage_len(self):
        return sum(1 for b in self.bit_coverage if b == '1')

    def add_rule(self, rule, cells):
        """
        Add a rule to the cell and update dependent cells based on rule's nonterminal pointers.
        Hiero cells are not added as dependents.
        :param rule: Rule object
        :param cells: Currents set of cells
        """

        # Ignore rules with empty target sides
        if len(rule.target_side) == 0:
            return

        self.rules.append(rule)
        self.carg_tokens.update(rule.carg_tokens)

        # Add dependent cells
        for _, nonterminal_coverage in rule.nonterminal_coverages.items():
            if nonterminal_coverage in cells:
                self.dependent_cells[nonterminal_coverage] = cells[nonterminal_coverage]

    def subtree(self, unexpanded_only=False):
        """
        Recursively compute a dictionary of int_coverage:Cell object pairs starting with current cell
        and including all dependent cells lower in the hierarchy.
        :param unexpanded_only: Only recurse into children if current RTN has not been expanded yet.
        :return: Dictionary of int_coverage:Cell object pairs
        """

        # Include current cell
        subtree_cells = {self.int_coverage: self}

        if unexpanded_only and self.expanded_rtn:
            return subtree_cells

        # Recursively add dependent cells
        for _, cell in self.dependent_cells.items():
            subtree_cells.update(cell.subtree())

        return subtree_cells

    def subtree_carg_tokens(self):
        """
        Gather a set of all carg tokens used in rules in current cell's subtree.
        :return: Set of integer CARG tokens
        """

        carg_tokens = set()

        for _, cell in self.subtree().items():
            carg_tokens.update(cell.carg_tokens)

        return carg_tokens


class Rule(object):

    def __init__(self, rule_id, target_side, feature_dict, coverage, hiero_intersection_rule=False):
        self.id = rule_id
        self.target_side = target_side
        self.feature_dict = helpers.compute_hsst_features(feature_dict) if feature_dict is not None else {}
        self.coverage = coverage
        self.hiero_intersection_rule = hiero_intersection_rule

        self.nonterminal_coverages = self.compute_rule_nonterminal_coverages(coverage)

        self.carg_tokens = set()

        if 'hsst_carg_terminal_rule' in self.feature_dict or 'hsst_carg_nonterminal_rule' in self.feature_dict:
            self.carg_tokens = {int(token) for token in target_side if not token.startswith('X')}

    @staticmethod
    def compute_rule_nonterminal_coverages(coverage):
        """
        Compute coverages of all non-terminal symbols of the rule
        :param coverage: Coverage tuple consisting of 0, 1 or non-terminal symbol (e.g. X_0)
        :return:
        """

        rule_nonterminals = set(x for x in coverage if x != 0 and x != 1)
        nonterminal_coverages = dict((nt, helpers.nt_coverage_to_int(coverage, nt)) for nt in rule_nonterminals)
        return nonterminal_coverages


def create_derivation_tree(sentence_grammar, sentence_coverage, hiero_peepholes=None, max_cell_coverage=0, allow_carg_rules=False):
    """
    Create derivation tree from sentence grammar and coverage.
    :param sentence_grammar: Dictionary of rule_id:rule tuple
    :param sentence_coverage: Dictionary of coverage:List of rule ids
    :param hiero_peepholes: Dictionary of HSST int coverage:Hiero int coverage pairs used for inclusion of additional rules.
    :param max_cell_coverage: Maximum number of nodes covered by a cell. Used for reducing the decoder workload.
     Default value of 0 does not limit the maximum size.
    :param allow_carg_rules: Exclude CARG rules if True
    :return: DerivationTree object
    """

    logging.info('Constructing derivation tree for the current sentence based on %d rules and %d coverages.' % (
        len(sentence_grammar) if sentence_grammar is not None else 0,
        len(sentence_coverage) if sentence_coverage is not None else 0
    ))

    coverage_cells = group_rules_by_coverage(
        sentence_grammar,
        sentence_coverage,
        allow_carg_rules=allow_carg_rules
    )

    derivation_tree = DerivationTree(hiero_peepholes=hiero_peepholes, max_cell_coverage=max_cell_coverage)

    for cell_int_coverage in sorted(coverage_cells, key=lambda x: int(x)):
        cell_rules = coverage_cells[cell_int_coverage]
        derivation_tree.add_cell(cell_int_coverage, cell_rules)

    logging.debug('Derivation tree with %d cells constructed.' % (len(derivation_tree.hsst_cells, )))

    if len(derivation_tree.hsst_cells) == 0:
        return derivation_tree

    subderivation_tree = create_top_cell_subtree(derivation_tree)

    return subderivation_tree


def create_top_cell_subtree(derivation_tree):
    """
    Create new derivation tree from existing one based on top hsst cell.
    :param derivation_tree: DerivationTree object
    :param space_intersection_func: Partial function for computing the intersections of HSST and Hiero space.
    :return: New DerivationTree object
    """

    hsst_cell_subset = derivation_tree.top_hsst_cell.subtree()

    subderivation_tree = DerivationTree(
        hiero_peepholes=derivation_tree.hiero_peepholes,
        max_cell_coverage=derivation_tree.max_cell_coverage
    )

    for int_coverage, cell in hsst_cell_subset.items():
        subderivation_tree.hsst_cells[int_coverage] = cell
        subderivation_tree.update_top_cell(cell)

    subderivation_tree.top_hsst_cell.is_top_cell = True

    return subderivation_tree


def create_hiero_intersection_subtree(derivation_tree, space_intersection_func):
    """
    Create new derivation tree from existing one based on space intersections between hiero and hsst.
    :param derivation_tree: DerivationTree object
    :param space_intersection_func: Partial function for computing the intersections of HSST and Hiero space.
     Only the set of cells in intersection subtrees is returned.
    :return: New DerivationTree object
    """

    # If space intersection function is defined, compute the intersections between HSST and Hiero
    space_intersections = space_intersection_func(derivation_tree.hsst_cells.keys())

    logging.info('Creating hiero intersection subset of all HSST cells based on %d space intersections.' % (
        len(space_intersections),
    ))

    for key, values in space_intersections.items():
        logging.debug("%s: %s" % (key, ' '.join(values)))

    hsst_cell_subset = dict()

    for hsst_int_coverage in sorted(space_intersections, key=lambda x: int(x)):
        intersected_cell = derivation_tree[hsst_int_coverage]
        intersected_cell.intersections = space_intersections[hsst_int_coverage]
        hsst_cell_subset.update(intersected_cell.subtree())

    subderivation_tree = DerivationTree(
        hiero_peepholes=derivation_tree.hiero_peepholes,
        max_cell_coverage=derivation_tree.max_cell_coverage
    )
    subderivation_tree.space_intersections = space_intersections

    for int_coverage, cell in hsst_cell_subset.items():
        subderivation_tree.hsst_cells[int_coverage] = cell
        subderivation_tree.update_top_cell(cell)

    return subderivation_tree


def group_rules_by_coverage(sentence_grammar, sentence_coverage, allow_carg_rules=False):
    """
    Group rules from sentence_grammar according to coverages in sentence_coverage.
    :param sentence_grammar: Dictionary of rule_id:rule tuple
    :param sentence_coverage: Dictionary of coverage:List of rule ids
    :param allow_carg_rules: Exclude CARG rules if True
    :return: Dictionary of int_coverage:List of Rule objects
    """

    logging.info('Grouping rules by coverage.')

    # Create a dictionary for storing grouped rules (coverage:rule_list)
    coverage_cells = defaultdict(list)

    # Return empty dictionary if there are no applied rules
    if sentence_grammar is None or sentence_coverage is None:
        return coverage_cells

    # Group rules according to their coverage
    for coverage, rule_ids in sentence_coverage.items():
        # Convert coverage with nonterminals into simple bit coverage
        int_coverage = helpers.coverage_to_int_string(coverage)

        for rule_id in rule_ids:
            # Retrieve rule by its rule_id from sentence_grammar
            rule_id, target_side, feature_dict = sentence_grammar[rule_id]

            if feature_dict['rule_type'].startswith('carg') and not allow_carg_rules:
                continue

            # Create rule object
            rule = Rule(rule_id, target_side, feature_dict, coverage)

            # Group rules by coverage
            coverage_cells[int_coverage].append(rule)

    return coverage_cells
