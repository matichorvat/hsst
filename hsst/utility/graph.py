import json
import copy
import logging
import xml.etree.ElementTree as xml


class SemanticGraph(object):

    def __init__(self, nodes, edges, nonterminal_count=0, ltop=None, index=None, attrib=None):
        self.subgraphs = set()
        self.nodes = set(nodes)
        self.edges = set(edges)
        self.nonterminal_count = nonterminal_count

        # If LTOP or index are None, choose random node
        node_dict = dict((node.node_id, node) for node in self.nodes)
        self.ltop = node_dict[ltop] if ltop is not None and ltop in node_dict else None  #next(iter(self.nodes))
        self.index = node_dict[index] if index is not None and index in node_dict else None  #next(iter(self.nodes))

        self.attrib = attrib

    def __str__(self):
        return 'Graph\n\tNodes: %s\n\tEdges: %s' % \
               (",".join(unicode(x.label) for x in sorted(self.nodes, key=lambda x: x.node_id)),
                "|".join(x.string for x in sorted(self.edges)))

    def __repr__(self):
        return "SemanticGraph(nodes={%r},edges={%r})" % \
               (",".join(str(x) for x in sorted(self.nodes)), ",".join(str(x) for x in sorted(self.edges)))

    def __hash__(self):
        return hash(self.__repr__())

    def __len__(self):
        return len(self.nodes)

    def __eq__(self, other):
        if isinstance(other, SemanticGraph):
            return (self.nodes == other.nodes) and (self.edges == other.edges)
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def get_outgoing_node_edges(self, node):
        return [edge for edge in self.edges if edge.from_node.node_id == node.node_id]

    def get_incoming_node_edges(self, node):
        return [edge for edge in self.edges if edge.to_node.node_id == node.node_id]

    def get_incident_edges(self, node):
        return self.get_outgoing_node_edges(node) + self.get_incoming_node_edges(node)

    def get_child_nodes(self, node, ignore_edge_to_self=True):
        return [edge.to_node for edge in self.get_outgoing_node_edges(node) if not ignore_edge_to_self or edge.to_node != node]

    def get_parent_nodes(self, node, ignore_edge_to_self=True):
        return [edge.from_node for edge in self.get_incoming_node_edges(node) if not ignore_edge_to_self or edge.from_node != node]

    def get_adjacent_nodes(self, node, ignore_edge_to_self=True):
        return self.get_child_nodes(node, ignore_edge_to_self) + self.get_parent_nodes(node, ignore_edge_to_self)

    def is_subgraph(self, other):
        # Checks if self is a subgraph of other:
        # True - self is a subgraph of other
        # False - self is not a subgraph of other

        return self.nodes.issubset(other.nodes) and self.edges.issubset(other.edges)

    def contains_directed_cycle(self):
        """
        Check whether the graph contains a directed cycle by iteratively removing nodes that either have no parents or no children.
         When no nodes can be removed from the remaining list, the graph has a cycle.
        :return: Set of Node objects in the cycle or empty set if no cycle exists.
        """

        nodes_from = dict((node, set(self.get_child_nodes(node))) for node in self.nodes)
        nodes_to = dict((node, set(self.get_parent_nodes(node))) for node in self.nodes)

        remaining_nodes = set(self.nodes)
        has_node_been_removed = True

        while has_node_been_removed:
            has_node_been_removed = False

            for node in remaining_nodes:
                # If node has no children, remove it
                if len(nodes_from[node] & remaining_nodes) == 0:
                    remaining_nodes.remove(node)
                    has_node_been_removed = True
                    break

                # If node has no parents, remove it
                elif len(nodes_to[node] & remaining_nodes) == 0:
                    remaining_nodes.remove(node)
                    has_node_been_removed = True
                    break

        return remaining_nodes

    def contains_undirected_cycle(self):
        """
        Check whether the graph contains an undirected cycle by iteratively removing nodes that have a single adjacent node.
         When no nodes can be removed from the remaining list, the graph has a cycle.
        :return: Set of Node objects in the cycle or empty set if no cycle exists.
        """

        adjacent_nodes = dict((node, set(self.get_adjacent_nodes(node))) for node in self.nodes)

        remaining_nodes = set(self.nodes)
        has_node_been_removed = True

        while has_node_been_removed:
            has_node_been_removed = False

            for node in remaining_nodes:
                # If node has a single adjacent node, remove it
                if len(adjacent_nodes[node] & remaining_nodes) <= 1:
                    remaining_nodes.remove(node)
                    has_node_been_removed = True
                    break

        return remaining_nodes

    def contains_cycle(self):
        """
        Checks whether the graph contains any cycle, directed or undirected.
        :return: Set of Node objects in the cycle or empty set if no cycle exists. Directed cycles are given priority.
        """
        directed_cycle = self.contains_directed_cycle()

        if directed_cycle:
            return directed_cycle

        undirected_cycle = self.contains_undirected_cycle()

        if undirected_cycle:
            return undirected_cycle

        return set()

    def find_disconnected_subgraphs(self):
        """
        Find parts of the graph disconnected from the main part (indicated by Index).
        :return: A list of disconnected graph parts, in the form of sets of Node objects.
        Returns empty list if graph is not disconnected.
        """

        disconnected_parts = []

        # Find nodes disconnected from the main part of the graph
        unvisited_nodes = set(self.nodes)

        # Initialize graph traversal
        if self.index is not None:
            start_node = self.index
        elif self.ltop is not None:
            start_node = self.ltop
        else:
            start_node = next(iter(self.nodes))

        explore_set = set(self.get_adjacent_nodes(start_node))
        unvisited_nodes.remove(start_node)

        # Iteratively visit a node and update the explore set with neighbouring nodes until explore set is empty
        while explore_set:
            node = explore_set.pop()
            unvisited_nodes.remove(node)
            explore_set.update(set(self.get_adjacent_nodes(node)) & unvisited_nodes)

        # If no nodes were unvisited, the graph has no disconnected parts
        if len(unvisited_nodes) == 0:
            return disconnected_parts

        # Collect disconnected parts into a list of parts by repeating the above procedure on unvisited nodes
        while unvisited_nodes:

            disconnected_part = set()

            # Initialize graph traversal
            start_node = next(iter(unvisited_nodes))
            explore_set = set(self.get_adjacent_nodes(start_node))
            unvisited_nodes.remove(start_node)
            disconnected_part.add(start_node)

            # Iteratively visit a node and update the explore set with neighbouring nodes until explore set empty
            while explore_set:
                node = explore_set.pop()
                unvisited_nodes.remove(node)
                disconnected_part.add(node)
                explore_set.update(set(self.get_adjacent_nodes(node)) & unvisited_nodes)

            disconnected_parts.append(disconnected_part)

        return disconnected_parts


class Node(object):

    def __init__(self, node_id, lemma=None, sense=None, pos=None, gpred=None, is_nonterminal=False,
                 label=None, tokalign=None, xml_entity=None):
        self.node_id = node_id
        self.lemma = lemma
        self.sense = sense
        self.pos = pos
        self.gpred = gpred
        self.label = label
        self.is_nonterminal = is_nonterminal
        self.tokalign = {} if tokalign is None else tokalign
        self.xml_entity = xml_entity

    def __str__(self):
        return str(self.node_id)

    def __repr__(self):
        return "Node(id=%r,label=%r)" % (self.node_id, self.label)

    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, other):
        if isinstance(other, Node):
            return (self.node_id == other.node_id) and (self.label == other.label)
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __cmp__(self, other):
        return cmp(self.node_id, other.node_id)


class Edge(object):

    def __init__(self, from_node, to_node, label, xml_entity=None):
        self.from_node = from_node
        self.to_node = to_node
        self.label = label
        self.xml_entity = xml_entity

    @property
    def edge_id(self):
        return "%s,%s,%s" % (self.from_node, self.to_node, self.label)

    @property
    def string(self):
        return "%s,%s,%s" % (self.from_node.label, self.to_node.label, self.label)

    def __str__(self):
        return str(self.edge_id)

    def __repr__(self):
        return "Edge(fromNode=%r,toNode=%r,label=%r)" % \
               (self.from_node, self.to_node, self.label)

    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, other):
        if isinstance(other, Edge):
            return (self.from_node == other.from_node) and \
                   (self.to_node == other.to_node) and \
                   (self.label == other.label)
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __cmp__(self, other):
        if cmp(self.from_node, other.from_node):
            return cmp(self.from_node, other.from_node)

        elif cmp(self.to_node, other.to_node):
            return cmp(self.to_node, other.to_node)

        elif cmp(self.label, other.label):
            return cmp(self.label, other.label)

        else:
            return 0


def standardize_nonterminals(graph, coverage_list):
    """
    Ensure that non-terminal labels X_0 and X_1 are assigned in a standardized fashion.
     If they are not, swap them both in the grap and in the list of target tokens.
     Note: Swapping modified the original graph.
    :param graph: Graph object
    :param coverage_list: Set of coverage tuples
    :return: SemanticGraph object, Set of coverage tuples
    """

    if graph.nonterminal_count != 2:
        return graph, coverage_list

    swapped = standardize_graph_nonterminals(graph)

    if not swapped:
        return graph, coverage_list

    swap_list = []
    for coverage in coverage_list:
        swap_list.append(tuple(swap_target_nonterminals(coverage)))

    return graph, set(swap_list)


def standardize_graph_nonterminals(graph):
    """
    Ensure that non-terminal node labels X_0 and X_1 are assigned in a standardized fashion. If they are not, swap the labels.
    :param graph: SemanticGraph object
    :return: Boolean if swap has occurred
    """

    # This method is only relevant if there are two non-terminals in the graph
    assert graph.nonterminal_count == 2

    # Get x0 and x1 nodes
    nonterminal_nodes = sorted([node for node in graph.nodes if node.is_nonterminal], key=lambda x: x.label)

    x0_node = nonterminal_nodes[0]
    x1_node = nonterminal_nodes[1]

    # Create a canonic representation for each of them
    x0_canonic_repr = canonic_nonterminal_node_key(x0_node, graph)
    x1_canonic_repr = canonic_nonterminal_node_key(x1_node, graph)

    # If they are not correctly ordered according to lexicographic sort, swap x0 and x1
    if x0_canonic_repr > x1_canonic_repr:
        swap_node_nonterminals(x0_node, x1_node)
        return True

    return False


def canonicalize_graph(graph):
    """
    Construct a canonic graph with a standard ordering of nodes and edges.
    """
    node_to_node_mapping = {}

    old_nodes = sorted(graph.nodes, key=lambda x: canonic_node_key(x, graph))

    new_nodes = set()
    new_edges = set()

    for node_id, node in enumerate(old_nodes):
        new_node = Node(
            node_id,
            lemma=node.lemma,
            sense=node.sense,
            pos=node.pos,
            gpred=node.gpred,
            is_nonterminal=node.is_nonterminal,
            label=node.label,
            xml_entity=node.xml_entity.copy()
        )

        new_node.xml_entity.attrib['nodeid'] = str(node_id + 1)

        new_nodes.add(new_node)
        node_to_node_mapping[node] = new_node

    for edge in graph.edges:
        new_edge = Edge(node_to_node_mapping[edge.from_node], node_to_node_mapping[edge.to_node], edge.label, xml_entity=edge.xml_entity.copy())

        new_edge.xml_entity.attrib['from'] = str(node_to_node_mapping[edge.from_node].node_id + 1)
        new_edge.xml_entity.attrib['to'] = str(node_to_node_mapping[edge.to_node].node_id + 1)

        new_edges.add(new_edge)

    return SemanticGraph(new_nodes, new_edges, graph.nonterminal_count + 1), node_to_node_mapping


def canonic_nonterminal_node_key(node, graph):
    """
    Construct a key for a non-terminal node in a graph that will be able to uniquely order it among non-terminal nodes.
    The key must not use non-terminal label ('X0', 'X1') since they are subject to swapping.
    :param node: Node object
    :param graph: SemanticGraph object
    :return: Canonic string representing node
    """
    # Get edges adjacent to node
    adjacent_edges = get_incident_edges(node, graph.edges)

    # Get nodes adjacent to node
    adjacent_nodes = [edge.from_node for edge, _ in adjacent_edges if edge.from_node != node]
    adjacent_nodes += [edge.to_node for edge, _ in adjacent_edges if edge.to_node != node]

    # Get edges one removed from node
    one_removed_edges = [get_incident_edges(adjacent_node, graph.edges, node) for adjacent_node in adjacent_nodes]
    one_removed_edges = [edge for edge_list in one_removed_edges for edge in edge_list]

    # Create lexical representations of adjacent edges:
    #   to_node is omitted since it equals non_terminal label
    #   is_incoming_edge boolean is used for additional differentiation in the key (case: 0_0_X0_X1|0-1-1_0-2-0_1-3-0)
    adjacent_edge_lex = sorted([get_edge_string(edge, is_incoming_edge, to_node=False) for edge, is_incoming_edge in adjacent_edges])

    # Create lexical representations of one removed edges - all information can be used here
    one_removed_edge_lex = sorted([get_edge_string(edge, is_incoming_edge) for edge, is_incoming_edge in one_removed_edges])

    # Join adjacent and one removed edge lexical information into a string
    canonic_representation = (adjacent_edge_lex, one_removed_edge_lex)
    return json.dumps(canonic_representation)


def canonic_node_key(node, graph):
    """
    Construct a key for a non-terminal node in a graph that will be able to uniquely order it in a set of nodes.
    :param node: Node object
    :param graph: SemanticGraph object
    :return: Canonic string representing node
    """
    # Get edges adjacent to node
    adjacent_edges = get_incident_edges(node, graph.edges)

    # Get nodes adjacent to node
    adjacent_nodes = [edge.from_node for edge, _ in adjacent_edges if edge.from_node != node]
    adjacent_nodes += [edge.to_node for edge, _ in adjacent_edges if edge.to_node != node]

    # Get edges one removed from node
    one_removed_edges = [get_incident_edges(adjacent_node, graph.edges, node) for adjacent_node in adjacent_nodes]
    one_removed_edges = [edge for edge_list in one_removed_edges for edge in edge_list]

    # Create lexical representations of adjacent edges:
    #   is_incoming_edge boolean is used for additional differentiation in the key
    adjacent_edge_lex = sorted([get_edge_string(edge, is_incoming_edge) for edge, is_incoming_edge in adjacent_edges])

    # Create lexical representations of one removed edges - all information can be used here
    one_removed_edge_lex = sorted([get_edge_string(edge, is_incoming_edge) for edge, is_incoming_edge in one_removed_edges])

    # Join node label, adjacent and one removed edge lexical information into a string
    canonic_representation = (node.label, adjacent_edge_lex, one_removed_edge_lex)
    return json.dumps(canonic_representation)


def get_incident_edges(node, edges, forbidden_node=None):
    """
    Get a list of incident edges to node, that do not neighbor the forbidden node.
    :param node: Node object
    :param edges: Collection of edge objects
    :param forbidden_node: Node object
    :return: List of tuples (Edge, is_incoming_edge) incident to node. is_incoming_edge is True if the edge is incoming
    to node, otherwise it's False
    """
    incoming_edges = [(edge, True) for edge in edges if edge.to_node == node]
    outgoing_edges = [(edge, False) for edge in edges if edge.from_node == node]

    if forbidden_node is not None:
        incoming_edges = filter(lambda edge: edge[0].from_node != forbidden_node, incoming_edges)
        outgoing_edges = filter(lambda edge: edge[0].to_node != forbidden_node, outgoing_edges)

    return incoming_edges + outgoing_edges


def get_edge_string(edge, is_incoming_edge, to_node=True):
    """
    Create string representing the edge.
    :param edge: Edge object
    :param is_incoming_edge: Boolean, True if edge is incoming from the perspective of a node
    :param to_node: Boolean, output both from and to node labels if True
    :return: String representing the edge
    """

    if to_node:
        return "%s,%s,%s,%d" % (edge.from_node.label, edge.to_node.label, edge.label, 1 if is_incoming_edge else 0)
    else:
        return "%s,%s,%d" % (edge.from_node.label, edge.label, 1 if is_incoming_edge else 0)


def swap_node_nonterminals(x0_node, x1_node):
    """
    Swap non-terminal nodes by changing their ids and labels.
    :param x0_node: Node object
    :param x1_node: Node object
    """
    x0_node.label, x1_node.label = x1_node.label, x0_node.label
    x0_node.node_id, x1_node.node_id = x1_node.node_id, x0_node.node_id
    x0_node.xml_entity, x1_node.xml_entity = x1_node.xml_entity, x0_node.xml_entity


def swap_target_nonterminals(target):
    """
    Swap non-terminal tokens.
    :param target: List of target tokens
    :return: List of target tokens
    """
    return ['X_1' if token == 'X_0' else 'X_0' if token == 'X_1' else token for token in target]


def read_graphs(sentence_collection, idx=True):
    """
    Read DMRS XML representation into SemanticGraphs for a sentence collection.
    """

    for sentence in sentence_collection:
        if sentence.source is not None and sentence.source.dmrs is not None:
            sentence.source.graph = read_graph(sentence.source.dmrs, idx)

        if sentence.target is not None and sentence.target.dmrs is not None:
            sentence.target.graph = read_graph(sentence.target.dmrs, idx)


def read_graph(dmrs, idx=True):
    """
    Load a DMRS XML graph representation into SemanticGraph object consisting of Nodes and Edges.
    """

    nodes = {}
    edges_raw = []

    for element in dmrs:

        if element.tag == 'node':
            node_id = element.attrib.get('nodeid')

            if idx:
                label = element.attrib.get('label_idx')
            else:
                label = element.attrib.get('label')

            tokalign = element.attrib.get('tokalign')

            if tokalign == '-1' or tokalign is None:
                tokalign = []
            else:
                tokalign = [int(tok) for tok in tokalign.split(' ') if tok != '']

            node = None

            if element.findall('realpred'):
                realpred = element.findall('realpred')[0] if element.findall('realpred') else None
                lemma = realpred.attrib.get('lemma')
                sense = realpred.attrib.get('sense')
                pos = realpred.attrib.get('pos')

                node = Node(node_id, lemma=lemma, sense=sense, pos=pos, label=label, tokalign=tokalign, xml_entity=element)

            elif element.findall('gpred'):
                gpred = element.findall('gpred')[0].text if element.findall('gpred') else None

                node = Node(node_id, gpred=gpred, label=label, tokalign=tokalign, xml_entity=element)

            nodes[node_id] = node

        elif element.tag == 'link':

            node_from = element.attrib.get('from')
            node_to = element.attrib.get('to')

            if idx:
                label = element.attrib.get('label_idx')
            else:
                label = element.attrib.get('label')

            edges_raw.append((node_from, node_to, label, element))

    edges = []
    for edge in edges_raw:
        try:
            edges.append(Edge(nodes[edge[0]], nodes[edge[1]], edge[2], xml_entity=edge[3]))
        except KeyError:
            # Skip, but really, those edges should have been removed in preprocessing
            logging.warn('Edge connecting non-existing node (%s, %s).' % (edge[0], edge[1]))
            continue

    return SemanticGraph(
        nodes.values(),
        edges,
        0,
        ltop=dmrs.attrib.get('ltop'),
        index=dmrs.attrib.get('index'),
        attrib=dict(dmrs.attrib)
    )


def write_graph(graph):
    dmrs_xml = xml.Element('dmrs')
    dmrs_xml.attrib = graph.attrib if graph.attrib is not None else {}
    dmrs_xml.text = '\n'
    dmrs_xml.tail = '\n'

    for node in sorted(graph.nodes):
        node.xml_entity.tail = '\n'
        dmrs_xml.append(node.xml_entity)

    for edge in sorted(graph.edges):
        edge.xml_entity.tail = '\n'
        dmrs_xml.append(edge.xml_entity)

    return dmrs_xml


def attach_nonterminal_xml_entity(node):
    xml_entity = xml.Element('node')
    gpred = xml.Element('gpred')
    gpred.text = node.label
    sortinfo = xml.Element('sortinfo')
    xml_entity.append(gpred)
    xml_entity.append(sortinfo)
    node.xml_entity = xml_entity
