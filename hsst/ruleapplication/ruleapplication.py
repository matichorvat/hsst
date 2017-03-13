import itertools
from copy import deepcopy
from collections import defaultdict

from hsst.utility.dfs_subgraph_enumeration import enumerate_dfs_subgraphs
from hsst.rulextraction.rulextraction import nonterminal_graph_edge_constraints
from hsst.utility.semantic_subgraph_enumeration import enumerate_semantic_parse_subgraphs
from hsst.utility.graph import Node, Edge, SemanticGraph, canonicalize_graph, standardize_nonterminals, attach_nonterminal_xml_entity


def ruleapplication(source_side, max_nonterminals=2, nodeset_size_limit=5, initial_nodeset_size_limit=0,
                    subgraph_df_limit=100, parse_graph=False):
    """
    Extract the maximum set of rules allowed under constraints from a single example consisting of source side graph.
    """

    # Compute reference coverage to be used for computing all other coverages
    reference_coverage = sorted([node.node_id for node in source_side.nodes])

    # Create a viable subgraphs from source_side graph and compute their coverages
    subgraph_coverages = create_subgraphs(source_side, reference_coverage, parse_graph, subgraph_df_limit)

    # Create a list of terminal graphs that will be used for subtraction in non-terminal graph creation
    subtraction_graphs = create_subtraction_graphs(subgraph_coverages.keys(), initial_nodeset_size_limit)

    # Create covereges dictionary for tracking coverages of all graphs
    coverages = defaultdict(set)
    coverages.update(subgraph_coverages)

    # Initialize the working set of graphs to all subgraphs
    current_graphs = subgraph_coverages.keys()

    # Construct nonterminal graphs, adding one nonterminal per iteration, until no more nonterminals can be added
    for iteration in range(0, max_nonterminals):
        nonterminal_graphs = set()

        # Iterate over graphs in the current operating set of graphs
        for graph in current_graphs:

            for subtraction_graph in subtraction_graphs:

                if len(subtraction_graph) >= len(graph):
                    # Stop iteration as all following subtraction_graphs will also be larger (they are sorted by size)
                    break

                if not subtraction_graph.is_subgraph(graph):
                    # If subtraction_graph is not a subgraph of graph, continue in the next iteration
                    continue

                # Attempt creating a nonterminal_graph and its coverages by subtracting subtraction_graph from graph
                nonterminal_graph, nonterminal_coverages = create_nonterminal_graph(graph, subtraction_graph,
                                                                                    coverages[graph],
                                                                                    next(iter(coverages[subtraction_graph])))

                if nonterminal_graph is not None:
                    # If nonterminal_graph was created, add it to current iteration graphs and all graphs
                    nonterminal_graphs.add(nonterminal_graph)
                    coverages[nonterminal_graph].update(nonterminal_coverages)

        # Filter out nonterminal graphs that cannot have any graph subtracted from it anymore
        current_graphs = filter(lambda x: len(x) - x.nonterminal_count >= 1, nonterminal_graphs)

    # Filter and canonicalize graphs
    filtered_graph_coverages = filter_graphs(coverages, nodeset_size_limit)
    canonicalized_graph_coverages = canonicalize_graphs(filtered_graph_coverages)

    # print len(canonicalized_graph_coverages)
    #
    # all_coverages = set()
    # for coverage_set in canonicalized_graph_coverages.values():
    #     for coverage in coverage_set:
    #         all_coverages.add(tuple('1' if x.startswith('X') else x for x in coverage))
    #
    # print len(all_coverages)

    return canonicalized_graph_coverages


def create_subgraphs(graph, reference_coverage, parse_graph=False, subgraph_df_limit=20):
    """
    Create a list of subgraphs from graph and compute their coverages.
    :param graph: SemanticGraph object
    :param reference_coverage: List of node ids
    :param parse_graph: Whether to use original graph parsing approach that enumerates all subgraphs. False by default.
    :param subgraph_df_limit: If parse_graph is true, this limits the depth of subgraph enumeration.
    :return: Dictionary of SemanticGraph objects mapped to a single element set containing a tuple node coverage
    """

    if not parse_graph:
        subgraphs = enumerate_semantic_parse_subgraphs(graph)

    # Enumerate all the subgraphs in the graph and take only valid subgraphs that conform to the argument constraint
    else:
        subgraphs = filter(lambda x: is_subgraph_valid(x, graph), enumerate_dfs_subgraphs(graph, subgraph_df_limit))

    # Compute subgraph graph coverages
    subgraph_coverages = dict()

    for subgraph in subgraphs:
        subgraph_coverages[subgraph] = {create_node_coverage(subgraph, reference_coverage)}

    return subgraph_coverages


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


def create_node_coverage(graph, reference_coverage):
    """
    Create node coverage for a terminal graph given a reference coverage consisting of node_ids.
    """

    coverage = ['0'] * len(reference_coverage)
    graph_node_ids = set(node.node_id for node in graph.nodes)

    for index, node_id in enumerate(reference_coverage):
        if node_id in graph_node_ids:
            coverage[index] = '1'

    return tuple(coverage)


def create_subtraction_graphs(subgraphs, subtraction_graph_nodeset_size_limit):
    """
    Filter and sort subgraphs to create a list of terminal graphs
    :param subgraphs: List of SemanticGraph objects
    :param subtraction_graph_nodeset_size_limit: Integer limiting the maximum node set size of terminal graphs
    :return: List of SemanticGraph objects
    """

    # Construct the set of terminal graphs that limits the size of subgraphs to     subtraction_graph_nodeset_size_limit
    if subtraction_graph_nodeset_size_limit > 0:
        terminal_graphs = filter(lambda x: len(x) <= subtraction_graph_nodeset_size_limit, subgraphs)
    else:
        terminal_graphs = list(subgraphs)

    # Sort the init_graphs so that later iteration can stop based on the length comparison
    terminal_graphs.sort(key=lambda x: len(x))

    return terminal_graphs


def create_nonterminal_graph(graph, subtraction_graph, graph_coverages, subtraction_coverage):
    """
    Create a non-terminal graph by subtracting the init_graph from graph. Node and Edge objects are deep copied in case
     they are modified at a later time.
    :param graph: SemanticGraph object
    :param subtraction_graph: SemanticGraph object
    :param graph_coverages: List of coverage tuples
    :param subtraction_coverage: Coverage tuple
    :return: SemanticGraph object
    """

    # Create the new nonterminal node with correct label
    node_id = 'X_%s' % str(graph.nonterminal_count)
    new_nonterminal_node = Node(node_id, label=node_id, is_nonterminal=True)

    if next(iter(graph.nodes)).xml_entity is not None:
        attach_nonterminal_xml_entity(new_nonterminal_node)

    # Take the difference of the two graphs
    difference_nodes = graph.nodes - subtraction_graph.nodes
    difference_edges = graph.edges - subtraction_graph.edges

    # Apply edge constraints, plus check that the graph does not consist purely of non-terminal nodes
    if len(difference_nodes) <= graph.nonterminal_count or not nonterminal_graph_edge_constraints(difference_edges, subtraction_graph):
        return None, None

    # Create deep copies of Node objects
    new_nodes_dict = {}
    for node in difference_nodes:
        new_nodes_dict[node.node_id] = deepcopy(node)

    # Replace initial graph nodes in the new graph edges with a nonterminal node or deep copy Edge object
    new_edges = set()
    for edge in difference_edges:

        # If edge terminates in the init_graph, replace its target node with new_nonterminal_node
        if edge.to_node in subtraction_graph.nodes:
            new_edges.add(Edge(new_nodes_dict[edge.from_node.node_id], new_nonterminal_node, edge.label, xml_entity=edge.xml_entity))

        else:
            new_edges.add(Edge(new_nodes_dict[edge.from_node.node_id], new_nodes_dict[edge.to_node.node_id], edge.label, xml_entity=edge.xml_entity))

    # Add the nonterminal node to nodeset
    new_nodes = set(new_nodes_dict.values())
    new_nodes.add(new_nonterminal_node)

    # Create new graph from the new node set and new edge set, with updated nonterminal count
    new_graph = SemanticGraph(new_nodes, new_edges, nonterminal_count=graph.nonterminal_count + 1)

    # Compute coverages for the new graph
    new_coverages = create_nonterminal_coverages(graph_coverages, subtraction_coverage, new_graph.nonterminal_count - 1)

    # Standardize non-terminals if needed
    new_graph, new_coverages = standardize_nonterminals(new_graph, new_coverages)

    return new_graph, new_coverages


def create_nonterminal_coverages(graph_coverages, subtraction_coverage, nonterminal_index):
    """
    Create nonterminal coverages by subtracting subtraction_coverage from all graph_coverages
    :param graph_coverages: List of coverage tuples
    :param subtraction_coverage: Coverage tuple
    :param nonterminal_index: Integer
    :return: List of subtracted coverage tuples
    """

    new_coverages = set()
    nonterminal = 'X_%d' % nonterminal_index

    for graph_coverage in graph_coverages:
        new_coverage = list()

        for graph_value, subgraph_value in itertools.izip(graph_coverage, subtraction_coverage):

            if graph_value == '1' and subgraph_value == '1':
                new_coverage.append(nonterminal)
            elif graph_value == '1' and subgraph_value == '0':
                new_coverage.append('1')
            elif graph_value == '0' and subgraph_value == '0':
                new_coverage.append('0')
            elif graph_value.startswith('X') and subgraph_value == '0':
                new_coverage.append(graph_value)
            else:
                raise Exception('The combination of coverage bits should not have occured.')

        new_coverages.add(tuple(new_coverage))

    return new_coverages


def filter_graphs(graph_coverages, nodeset_size_limit):
    """
    Filter graphs if they exceed nodeset_size_limit.
    :param graph_coverages: Dictionary of SemanticGraph objects mapped to sets of coverage tuples
    :param nodeset_size_limit: Maximum size of graph node set
    :return: Filtered dictionary of SemanticGraph objects mapped to sets of coverage tuples
    """

    if nodeset_size_limit > 0:
        # The maximum nodeset size constraint is applied after all rules are created
        # This is to prevent restricting the space of possible rules

        final_graphs = filter(lambda x: len(x) <= nodeset_size_limit, graph_coverages.keys())

        return dict((graph, graph_coverages[graph]) for graph in final_graphs)

    else:
        return graph_coverages


def canonicalize_graphs(graph_coverages):
    """
    Canonicalize graphs. New SemanticGraph objects are created.
    :param graph_coverages: Dictionary of SemanticGraph objects mapped to sets of coverage tuples
    :return: Dictionary of SemanticGraph objects mapped to sets of coverage tuples
    """

    canonic_graph_coverages = defaultdict(set)
    for graph, coverages in graph_coverages.items():
        canonicalized_graph, _ = canonicalize_graph(graph)
        canonic_graph_coverages[canonicalized_graph].update(coverages)

    return canonic_graph_coverages
