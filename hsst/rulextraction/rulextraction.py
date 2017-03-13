import logging
from copy import deepcopy

from hsst.utility import utility
from hsst.rulextraction.rule import Rule
from hsst.utility.graph import Node, Edge, SemanticGraph, attach_nonterminal_xml_entity
from hsst.utility.dfs_subgraph_enumeration import enumerate_dfs_subgraphs
from hsst.utility.semantic_subgraph_enumeration import enumerate_semantic_parse_subgraphs


def rulextraction(source_side, target_side, alignment, max_nonterminals=2, nodeset_size_limit=5,
                 initial_rule_nodeset_size_limit=0, subgraph_df_limit=100, parse_graph=False):
    """
    Extract the maximum set of rules allowed under constraints from a single example consisting of source side graph,
    target side string, and the alignment between them.
    """

    rules, subtraction_rules = create_terminal_and_subtraction_rules(
        source_side,
        target_side,
        alignment,
        initial_rule_nodeset_size_limit,
        parse_graph,
        subgraph_df_limit
    )

    logging.debug('Constructed %d initial rules and a total of %d rules in the first stage' %
                  (len(subtraction_rules), len(rules)))

    # Construct nonterminal rules, adding one nonterminal per iteration, until no more nonterminals can be added
    current_rules = rules

    for iteration in range(0, max_nonterminals):
        new_rules = set()

        # Iterate over rules in the current operating set of rules
        # (in the first iteration it consists of all terminal rules respecting constraints)

        for rule in current_rules:

            # Ensure the maximum number of nonterminals in a rule constraint
            if 0 < max_nonterminals <= rule.nonterminal_count:
                continue

            # Check if any initial rule is this rule's subgraph
            for init_rule in subtraction_rules:

                # If the initial rule is a subgraph, ensure that it is a proper subgraph
                # (the graphs are not identical and there is at least one token in the target string)
                if len(init_rule.source_side.nodes) >= len(rule.source_side.nodes) or len(rule.target_side) == 0:
                    continue

                # If the init_rule's source side is not a subgraph of rule's source side or init_rule's target side is not a substring of rule's target side
                if not init_rule.source_side.is_subgraph(rule.source_side) or not init_rule.is_substring(rule, alignment):
                    continue

                # If the basic constraints are satisfied, create a new nonterminal rule by subtracting init_rule from rule
                nonterminal_rule = create_nonterminal_rule(rule, init_rule, target_side, alignment)

                # Check that it respects the constraints
                if nonterminal_rule is None or not nonterminal_rule_constraints(nonterminal_rule):
                    continue

                new_rules.add(nonterminal_rule)

        # The operating set of rules in the next iteration are the newly constructed rules that can have a graph subtracted from it
        current_rules = filter(lambda x: len(x.source_side) - x.source_side.nonterminal_count >= 1, new_rules)

        logging.debug('Found %d new rules in iteration %d' % (len(new_rules), iteration))

        # Add new rules to the complete rule set with a union operation
        rules |= new_rules

    logging.debug('Found %d total rules' % (len(rules)))

    # Filter and canonicalize graphs
    filtered_rules = filter_rules(rules, nodeset_size_limit)
    logging.debug('Left with %d rules after size filtering' % (len(filtered_rules)))

    canonicalized_rules = canonicalize_rules(filtered_rules)

    return canonicalized_rules


def create_terminal_and_subtraction_rules(graph, target_side, alignment, subtraction_graph_nodeset_size_limit,
                                          parse_graph=False, subgraph_df_limit=20):
    """
    Create two sets of rules:
     1. The set of all terminal_rules whose subgraph, substring, and alignment conform to constraints
     2. Its subset, subtraction_rules, that limits the size of subgraphs to initial_rule_nodeset_size_limit
    :param graph: SemanticGraph object
    :param target_side: List of target side tokens
    :param alignment: Dictionary mapping from Node objects to lists of tokens
    :param subtraction_graph_nodeset_size_limit: Integer limiting the maximum node set size of terminal graphs
    :param parse_graph: Whether to use original graph parsing approach that enumerates all subgraphs. False by default.
    :param subgraph_df_limit: If parse_graph is true, this limits the depth of subgraph enumeration.
    :return: Tuple of sets: terminal_rules and subtraction_rules
    """

    if not parse_graph:
        subgraphs = enumerate_semantic_parse_subgraphs(graph, ignore_if_cycle=True)

    # Enumerate all the subgraphs in the graph and take only valid subgraphs that conform to the argument constraint
    else:
        subgraphs = filter(lambda x: is_subgraph_valid(x, graph), enumerate_dfs_subgraphs(graph, subgraph_df_limit))

    logging.debug('Found %d subgraphs.' % (len(subgraphs), ))

    terminal_rules = set()
    subtraction_rules = set()

    for subgraph in subgraphs:

        # Construct the initial rule and rule_target_index, for checking initial_rule_constraints
        rule, rule_target_indexes = create_initial_rule(subgraph, target_side, alignment)

        # Check if the initial rule conforms to the constraints
        if not initial_rule_constraints(rule, alignment, rule_target_indexes):
            continue

        terminal_rules.add(rule)

        # If the rule is below initial_rule_nodeset_size_limit or the value for it is not set (0),
        # add the rule to subtraction_rules

        if 0 < subtraction_graph_nodeset_size_limit < len(subgraph.nodes):
            continue

        subtraction_rules.add(rule)

    return terminal_rules, subtraction_rules


def is_subgraph_valid(subgraph, graph):
    """
    Check whether a subgraph is valid, i.e. all outgoing edges of present nodes are included
    in some capacity.
    """

    for subgraph_node in subgraph.nodes:
        graph_node_outgoing_edges = filter(lambda edge: edge.from_node == subgraph_node, graph.edges)
        subgraph_node_outgoing_edges = filter(lambda edge: edge.from_node == subgraph_node, subgraph.edges)

        if len(graph_node_outgoing_edges) != len(subgraph_node_outgoing_edges) or not edge_match(
                graph_node_outgoing_edges, subgraph_node_outgoing_edges):
            return False

    return True


def edge_match(graph_edges, subgraph_edges):
    """
    Make sure the two edge sets match in terms of edge labels.
    """

    subgraph_edge_labels = [edge.label for edge in subgraph_edges]

    for graph_edge in graph_edges:
        if graph_edge.label in subgraph_edge_labels:
            subgraph_edge_labels.remove(graph_edge.label)
        else:
            return False

    return True


def create_initial_rule(subgraph, target_side, alignment):
    """
    Create an initial rule based on the subgraph.
    """

    # Create new (index-unadjusted) alignment and target side index list
    new_alignment = {}
    new_target_side_indexes = set()

    for node in alignment:
        if node in subgraph.nodes:
            new_alignment[node] = list(alignment[node])
            new_target_side_indexes.update(alignment[node])

    # Create a set of token indexes that are not aligned to the nodes in the current subgraph,
    # but are aligned to a node outside of it

    outside_token_indexes = set()
    for node in alignment:
        if node not in subgraph.nodes:
            outside_token_indexes.update(alignment[node])

    # Sort the index-unadjusted target side index list
    new_target_side_indexes = sorted(new_target_side_indexes)
    
    # Iterate over the sorted pairs of the index list and store unaligned tokens between them.
    # !!NB!! this procedure is responsible for occasionally omitting unaligned words
    new_unaligned_indexes = list()

    for prev_index, index in utility.pairwise(new_target_side_indexes):
        diff = index - prev_index

        if diff >= 2:
            for i in range(prev_index + 1, index):
                if i not in outside_token_indexes:
                    new_unaligned_indexes.append(i)
                    
    # Extend the index-unadjusted target side index list with unaligned token indexes and resort
    new_target_side_indexes.extend(new_unaligned_indexes)
    new_target_side_indexes = sorted(new_target_side_indexes)

    # Based on the index-unadjusted target side index list construct the target side of the rule
    new_target_side = [target_side[index] for index in new_target_side_indexes]
    
    # Adjust the alignment to the new target side of the rule
    fixed_new_alignment = dict()
    for node in new_alignment:
        fixed_new_alignment[node] = [new_target_side_indexes.index(x) for x in new_alignment[node]]

    # Adjust the unaligned set to the new target side of the rule
    new_unaligned_indexes = set([new_target_side_indexes.index(x) for x in new_unaligned_indexes])

    # Construct the initial rule
    rule = Rule('X', subgraph, new_target_side, fixed_new_alignment, new_unaligned_indexes, 0)

    return rule, set(new_target_side_indexes)


def initial_rule_constraints(rule, alignment, rule_target_indexes):
    """
    1. Check that the source side contains at least one node and that target side contains at least one token
    2. Check that no node outside of the subgraph connects to a token inside the aligned substring.

    The constraint that no word outside of the substring connects to the subgraph is ensured due
    to the fact that substring is built based on subgraph alignments.
    """

    # Check that the source side contains at least one node and that target side contains at least one token
    if len(rule.source_side.nodes) == 0 or len(rule.target_side) == 0:
        return False

    # Check that no node outside of the subgraph connects to a token inside the aligned substring
    outside_token_indexes = set()
    for node in alignment:
        if node not in rule.source_side.nodes:
            outside_token_indexes.update(alignment[node])
            if any([True if token_index in rule_target_indexes else False for token_index in alignment[node]]):
                return False

    # Check if the substring is contiguous by iterating over sorted substring and comparing indexes
    for prev_index, index in utility.pairwise(sorted(rule_target_indexes)):
        diff = index - prev_index

        if diff >= 2:
            for i in range(prev_index + 1, index):
                if i in outside_token_indexes:
                    # If there are one or mode indexes missing between two contiguous indexes of the substring,
                    # check whether they are aligned to other nodes. If yes, the string is not contiguous. If no,
                    # the missing indexes are simply unaligned
                    return False
                
    return True


def create_nonterminal_rule(rule, init_rule, target_side, alignment):
    """
    Create a new nonterminal rule by subtracting the init_rule from rule.
    """

    # Create the new nonterminal node with correct label
    node_id = 'X_%s' % str(rule.source_side.nonterminal_count)
    new_nonterminal_node = Node(node_id, label=node_id, is_nonterminal=True)

    if next(iter(rule.source_side.nodes)).xml_entity is not None:
        attach_nonterminal_xml_entity(new_nonterminal_node)

    # Take the difference of the two graphs
    difference_nodes = rule.source_side.nodes - init_rule.source_side.nodes
    difference_edges = rule.source_side.edges - init_rule.source_side.edges

    # Apply edge constraints, plus check that the graph does not consist purely of non-terminal nodes
    if len(difference_nodes) <= rule.source_side.nonterminal_count or not nonterminal_graph_edge_constraints(difference_edges, init_rule.source_side):
        return None

    # Create deep copies of Node objects
    new_nodes_dict = {}
    for node in difference_nodes:
        new_nodes_dict[node.node_id] = deepcopy(node)

    # Replace initial graph nodes in the new graph edges with a nonterminal node or deep copy Edge object
    new_edges = set()
    for edge in difference_edges:

        # If edge terminates in the init_graph, replace its target node with new_nonterminal_node
        if edge.to_node in init_rule.source_side.nodes:
            new_edges.add(Edge(new_nodes_dict[edge.from_node.node_id], new_nonterminal_node, edge.label, xml_entity=edge.xml_entity))

        else:
            new_edges.add(
                Edge(new_nodes_dict[edge.from_node.node_id], new_nodes_dict[edge.to_node.node_id], edge.label, xml_entity=edge.xml_entity))

    new_nodes = set(new_nodes_dict.values())

    # Create new (index-unadjusted) alignment and target side index list
    # Also keep track of target side indexes removed due to init_rule subtraction

    new_alignment = {}
    new_target_side_indexes = set()
    removed_target_side_indexes = set()

    for old_node in rule.source_side.nodes:
        if old_node in difference_nodes:
            new_node = new_nodes_dict[old_node.node_id]
            new_alignment[new_node] = list(rule.alignment[old_node])
            new_target_side_indexes.update(rule.alignment[old_node])
        else:
            removed_target_side_indexes.update(rule.alignment[old_node])

    # Sort the index-unadjusted target side index list
    new_target_side_indexes = sorted(new_target_side_indexes)

    # Compute the minimum subtracted index from init_rule so that we can insert a nonterminal in its position
    min_removed_index = min(removed_target_side_indexes)

    init_rule_unaligned_indexes = set(index + min_removed_index for index in init_rule.unaligned)
    new_unaligned_indexes = set(rule.unaligned) - init_rule_unaligned_indexes

    # Extend the index-unadjusted target side index list with unaligned token indexes and an index for nonterminal symbol
    new_target_side_indexes.extend(new_unaligned_indexes)
    new_target_side_indexes.append(min_removed_index)

    # Resort the index-unadjusted target side index list
    new_target_side_indexes = sorted(new_target_side_indexes)

    # Based on the index-unadjusted target side index list construct the target side of the rule
    new_target_side = [rule.target_side[index] if index != min_removed_index else new_nonterminal_node.label for index in
                       new_target_side_indexes]

    # Adjust the alignment to the new target side of the rule
    fixed_new_alignment = dict()
    for node in new_alignment:
        fixed_new_alignment[node] = [new_target_side_indexes.index(x) for x in new_alignment[node]]

    fixed_new_unaligned_indexes = [new_target_side_indexes.index(x) for x in new_unaligned_indexes]

    # Add new nonterminal node to nodeset
    new_nodes.add(new_nonterminal_node)

    # Add alignment between nonterminal node in source side graph and nonterminal symbol in target side string
    fixed_new_alignment[new_nonterminal_node] = [new_target_side.index(new_nonterminal_node.label)]

    # Construct new SemanticGraph
    new_graph = SemanticGraph(new_nodes, new_edges, nonterminal_count=rule.source_side.nonterminal_count + 1)

    # Construct the nonterminal rule
    new_rule = Rule(rule.lhs, new_graph, new_target_side, fixed_new_alignment, fixed_new_unaligned_indexes, rule.nonterminal_count + 1)

    # Standardize non-terminals if needed
    new_rule.standardize_nonterminals()

    return new_rule


def nonterminal_graph_edge_constraints(new_edges, init_source_side):
    """
    Ensure that the new non-terminal rule respects edge constraints:
        1. No edges should originate from subtracted part of the graph
        2. All edges in the new graph should connect to no more than one distinct node in the subtracted part of the graph
    """

    # Prevent rules with edges starting in the subtracted graph
    if len(set([edge.from_node for edge in new_edges if edge.from_node in init_source_side.nodes])) > 0:
        return False

    # Prevent rules with edges connecting to more than one different node in the subtracted graph
    if len(set([edge.to_node for edge in new_edges if edge.to_node in init_source_side.nodes])) > 1:
        return False

    return True


def nonterminal_rule_constraints(rule):
    """
    Check if the nonterminal rule respects all constraints:
        1. Require at least one terminal token on target side
        # 2. Prevent two or more non-terminals to be adjacent on the target side
        3. Non-adjacent non-terminal nodes in the source graph-fragment
    """

    # Eequire at least one terminal token on target side
    if len([x for x in rule.target_side if not x.startswith('X_')]) == 0:
        return False

    # Prevent two or more non-terminals to be adjacent on the target side
    # prev_nonterminal = False
    # for token in rule.target_side:
    #     if token.startswith('X_'):
    #         if prev_nonterminal:
    #             return False
    #         else:
    #             prev_nonterminal = True
    #     elif prev_nonterminal:
    #         prev_nonterminal = False

    # Non-adjacent non-terminal nodes in the source graph-fragment
    for edge in rule.source_side.edges:
        # Just preventing a non-terminal node to be an origin of an edge (which is a previous constraint) is enough
        if edge.from_node.is_nonterminal:
            return False

            # if edge.from_node.is_nonterminal in nonterminals and edge.to_node.is_nonterminal in nonterminals:
            # return False

    return True


def filter_rules(rules, nodeset_size_limit):
    """
    Filter rules if their graphs exceed nodeset_size_limit.
    :param rules: Set of Rule objects
    :param nodeset_size_limit: Maximum size of graph node set
    :return: Filtered set of Rule objects
    """

    if nodeset_size_limit > 0:
        # The maximum nodeset size constraint is applied after all rules are created
        # This is to prevent restricting the space of possible rules

        return filter(lambda x: len(x.source_side) <= nodeset_size_limit, rules)

    else:
        return rules


def canonicalize_rules(rules):
    """
    Canonicalize rules. New Rule objects are created.
    :param rules: Set of Rule objects
    :return: Set of Rule objects
    """

    canonicalized_rules = set()
    for rule in rules:
        canonic_rule = rule.canonicalize()
        canonicalized_rules.add(canonic_rule)

    return canonicalized_rules