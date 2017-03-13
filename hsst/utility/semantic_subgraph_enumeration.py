import xml.etree.ElementTree as xml

from hsst.utility.graph import SemanticGraph, write_graph


def print_subgraph(subgraph):
    print '; '.join(x.xml_entity.attrib['label'] for x in subgraph.nodes)
    dmrs_string = xml.tostring(write_graph(subgraph), encoding='utf-8')
    print dmrs_string.strip()


def enumerate_semantic_parse_subgraphs(graph, ignore_if_cycle=False):
    """
    Create subgraphs of graph according to the semantic parse. The recursive algorithm is sensitive to cycles, so they
     are broken ahead of time or it returns empty list of subgraphs
    :param graph: SemanticGraph object
    :param ignore_if_cycle: If True and graph contains a cycle, return empty set
    :return: List of subgraph SemanticGraph objects
    """

    if ignore_if_cycle and graph.contains_cycle():
        return set()

    if graph.index is not None:
        top_node = graph.index

    elif graph.ltop is not None:
        top_node = graph.ltop

    else:
        top_node = next(iter(graph.nodes))

    subgraphs = create_subgraphs(graph, top_node, [])

    # If there's disconnected parts of the graph, create subgraphs for them too
    disconnected_graph_parts = graph.find_disconnected_subgraphs()

    if disconnected_graph_parts:
        for disconnected_graph_part in disconnected_graph_parts:
            random_start_node = next(iter(disconnected_graph_part))
            disconnected_subgraphs = create_subgraphs(graph, random_start_node, [])
            subgraphs.update(disconnected_subgraphs)

    # print '*' * 50
    # for subgraph in subgraphs:
    #     print_subgraph(subgraph)

    return subgraphs


def create_subgraphs(graph, top_node, previous_top_nodes):
    """
    Recursively create subgraphs with top_node as starting point.
    :param graph: SemanticGraph object
    :param top_node: Node object
    :param previous_top_nodes: List of Node objects
    :return: Set of subgraph SemanticGraph objects
    """

    # print 'Creating subgraphs for:', top_node.lemma if top_node.gpred is None else top_node.gpred
    # print 'Previous top nodes:',
    # print '; '.join(x.xml_entity.attrib['label'] for x in previous_top_nodes)

    subgraphs = set()

    # Get a list of parents of top_node that have not had their subgraphs created yet
    parents = list(set(graph.get_parent_nodes(top_node)) - set(previous_top_nodes))

    if len(parents) > 0:
        # If any such parents exist, do not create a subgraph for the current top_node
        # It will be created as a child of its parents

        # Determine the top parents of the current top_node and create subgraphs with top parents as top_node instead
        # If multiple parents are returned, their parse will be ambiguous, as all permutations of their subgraphs will
        # be created

        for new_top_node in determine_top_parent(graph, parents):
            subgraphs.update(create_subgraphs(graph, new_top_node, previous_top_nodes))

    else:
        # If no such parents exist, create a subgraph by traversing accessible nodes without crossing nodes that have
        # previously been top nodes
        current_subgraph = create_subgraph(graph, top_node, previous_top_nodes)
        subgraphs.add(current_subgraph)

        # print 'Subgraph:',
        # print_subgraph(current_subgraph)

        # Recursively create subgraphs for all current top_node children, with top_node added as a previous top node
        children = graph.get_child_nodes(top_node)
        for node in children:
            subgraphs.update(create_subgraphs(graph, node, list(previous_top_nodes) + [top_node]))

    return subgraphs


parent_pos_priority = ['subord', 'c_v', 'v', 'c_p', 'p_v', 'p', 'c', 'c_n', 'p_n', 'c_q', 'q', 'a']


def determine_top_parent(graph, parents):
    """
    Determine which parent(s) should be the next top_node. In case multiple parents could be the next top_node, all of
     them are returned.
    :param graph: SemanticGraph object
    :param parents: List of Node objects
    :return: List of Node object
    """

    # If only one parent exists, return that one
    if len(parents) == 1:
        return [parents.pop()]

    # If multiple parents exist, use their part-of-speech tags to determine priority
    # If multiple parents with the same POS exist, they are returned as a list

    # print 'Parents:',
    # print_list_of_nodes(parents)

    priorities = []
    for parent_node in parents:
        parent_pos = determine_node_pos(graph, parent_node)

        if parent_pos is not None and parent_pos in parent_pos_priority:
            parent_node_priority = parent_pos_priority.index(parent_pos)

        else:
            parent_node_priority = len(parent_pos_priority) + 1

        priorities.append(parent_node_priority)

    candidate_tops = locate_min(priorities)

    return [parents[index] for index in candidate_tops]


def determine_node_pos(graph, node):
    """
    Determine POS tag for node. In case of conjunction and preposition, the POS tags of their children are taken
     into account to produce special tags for e.g. conjunction of verbs vs. conjunction of nouns.
    :param graph: SemanticGraph object
    :param node: Node object
    :return: POS tag string
    """

    if node.gpred is not None:
        return node.gpred

    elif node.pos not in {'c', 'p'}:
        return node.pos

    children_pos = []
    for child in graph.get_child_nodes(node):
        children_pos.append(determine_node_pos(graph, child))

    if node.pos == 'c' and len(set(x[0] for x in children_pos if x is not None)) == 1:
        return 'c_%s' % children_pos[0][0]

    elif node.pos == 'p':
        if 'v' in children_pos:
            return 'p_v'

        elif 'n' in children_pos:
            return 'p_n'

        else:
            return 'p'

    return node.pos


def create_subgraph(graph, top_node, previous_top_nodes=None):
    """
    Create a subgraph by traversing graph edges starting from top_node, and not traversing edges that would take them to
     any of previous_top_nodes.
    :param graph: SemanticGraph object
    :param top_node: Node object
    :param previous_top_nodes: List of Node objects
    :return: Subgraph SemanticGraph object
    """

    # Current top_node is always in the subgraph
    visited_node_set = {top_node}

    # previous_top_nodes are marked as visited so that they are ignored during traversal
    if previous_top_nodes is not None:
        visited_node_set.update(previous_top_nodes)

    # The starting explore_set consists of adjacent nodes of top_node without previously visited nodes
    explore_set = set(graph.get_adjacent_nodes(top_node)) - visited_node_set

    # Explore set is iteratively updated after visiting the nodes on the previous one
    while len(explore_set) > 0:
        candidate_explore_set = set()

        for node in explore_set:
            # Add node to visited nodes and extend candidate explore set with nodes adjacent to it
            visited_node_set.add(node)
            candidate_explore_set.update(graph.get_adjacent_nodes(node))

        # Previously visited nodes are ignored
        explore_set = candidate_explore_set - visited_node_set

    # The subgraph node set consists of visited nodes with previous_top_nodes removed
    if previous_top_nodes is not None:
        nodes = visited_node_set - set(previous_top_nodes)
    else:
        nodes = visited_node_set

    # The subgraph edge set consists of all edges that both start and terminate in subgraph nodes
    edges = set()
    for node in nodes:
        for edge in graph.get_incident_edges(node):
            if edge.from_node in nodes and edge.to_node in nodes:
                edges.add(edge)

    return SemanticGraph(nodes, edges, nonterminal_count=0)


def locate_min(a):
    """
    Get list of indexes of all minimum value elements of a.
    :param a: Iterable
    :return: List of indexes
    """
    smallest = min(a)
    return [index for index, element in enumerate(a) if smallest == element]
