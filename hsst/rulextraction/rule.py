from hsst.utility.graph import canonicalize_graph, standardize_graph_nonterminals


class Rule(object):

    def __init__(self, lhs, source_side, target_side, alignment, unaligned_set, nonterminal_count):
        self.lhs = lhs
        self.source_side = source_side
        self.target_side = list(target_side)
        self.alignment = alignment
        self.unaligned = unaligned_set
        self.nonterminal_count = nonterminal_count

    def __repr__(self):
        return "Rule(lhs=%r,source_side={%r},target_side={%r},alignment={%r},unaligned={%r})" % \
               (self.lhs,
                self.source_side,
                self.target_side,
                [(x, self.alignment[x]) for x in sorted(self.alignment.keys())],
                [x for x in sorted(self.unaligned)])

    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, other):
        if isinstance(other, Rule):
            return (self.lhs == other.lhs) and \
                   (self.source_side == other.source_side) and \
                   (self.target_side == other.target_side) and \
                   (self.alignment == other.alignment) and \
                   (self.unaligned == other.unaligned)
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return '%s ->  <\nSource side:\n%s\nTarget side:\n\t%s\nAlignment:\n\t%s\nUnaligned:\n\t%s\n>' % \
               (self.lhs,
                self.source_side,
                self.target_side,
                [(x.label, self.alignment[x]) for x in sorted(self.alignment.keys())],
                [x for x in sorted(self.unaligned)])

    def __cmp__(self, other):
        return len(self.source_side).__cmp__(len(other.source_side))

    def standardize_nonterminals(self):
        """
        Ensure that non-terminal labels X_0 and X_1 are assigned in a standardized fashion.
         If they are not, swap them both in the graph and in the list of target tokens.
         Note: Swapping modified the original graph.
        :param graph: Graph object
        :param target_list: List of lists of target tokens
        :param alignment: Alignment dictionary
        :return: SemanticGraph object, List of lists of target tokens
        """

        if self.source_side.nonterminal_count != 2:
            return

        # Store alignment information in case nonterminal nodes get swapped
        nonterminal_nodes = sorted([node for node in self.source_side.nodes if node.is_nonterminal],
                                   key=lambda x: x.label)
        old_x0_node = nonterminal_nodes[0]
        old_x1_node = nonterminal_nodes[1]
        old_x0_alignment = self.alignment.pop(old_x0_node, None)
        old_x1_alignment = self.alignment.pop(old_x1_node, None)

        # Swap nodes based on graph information
        swapped = standardize_graph_nonterminals(self.source_side)

        # If nodes were not swapped, reset alignment information
        if not swapped:
            self.alignment[old_x0_node] = old_x0_alignment
            self.alignment[old_x1_node] = old_x1_alignment
            return

        # If they were, retrieve new nonterminal nodes and swap alignment information
        nonterminal_nodes = sorted([node for node in self.source_side.nodes if node.is_nonterminal],
                                   key=lambda x: x.label)
        new_x0_node = nonterminal_nodes[0]
        new_x1_node = nonterminal_nodes[1]
        self.alignment[new_x1_node] = old_x0_alignment
        self.alignment[new_x0_node] = old_x1_alignment

        # Swap target side nonterminals
        self.target_side = tuple(swap_target_nonterminals(self.target_side))

    def canonicalize(self):
        canonic_graph, node_to_node_mapping = canonicalize_graph(self.source_side)
        canonic_alignment = {}

        for node in self.alignment:
            canonic_alignment[node_to_node_mapping[node]] = self.alignment[node]

        return Rule(self.lhs, canonic_graph, self.target_side, canonic_alignment, self.unaligned, self.nonterminal_count)

    def is_substring(self, rule, alignment):
        """
        Determine whether current rule's target side is a substring of rule's target side.
        :param rule: Rule object
        :param alignment: Sentence alignment dictionary (Node object: list of sentence indexes)
        :return: True if substring
        """

        # Obtain (ordered) lists of aligned sentence target indexes for current rule and rule
        init_indexes = sorted(set(index for node in self.alignment for index in alignment[node]))
        rule_indexes = sorted(set(index for node in rule.alignment if node in alignment for index in alignment[node]))

        # Try to find a subsequence init_indexes in rule_indexes
        for i in xrange(len(rule_indexes) - len(init_indexes) + 1):
            # Attempt to match init_indexes to rule_indexes from ith rule_index onwards
            for j in xrange(len(init_indexes)):
                if rule_indexes[i + j] != init_indexes[j]:
                    # If an element does not match, break checking start from ith element
                    break
            else:
                # If entire sequence was matched without breaking, return True
                return True

        return False


def swap_target_nonterminals(target):
    """
    Swap non-terminal tokens.
    :param target: List of target tokens
    :return: List of target tokens
    """
    return ['X_1' if token == 'X_0' else 'X_0' if token == 'X_1' else token for token in target]


