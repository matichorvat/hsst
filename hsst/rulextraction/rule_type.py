
def load_wmap(filename):
    wmap = {}
    with open(filename) as f:
        for line in f:
            line = line.strip()

            if line != '':
                line = line.split('\t')
                idx = line[0]
                symbol = line[1].decode('utf-8')

                wmap[idx] = symbol

    return wmap


def get_node_pos(node):
    if len(node.split('_')) >= 3:
        return node.split('_')[2]
    else:
        return ''


def get_top_nodes(nodes, edges):
    top_nodes = list(nodes)

    for edge in edges:
        node_in = nodes[int(edge[1])]

        if node_in in top_nodes:
            top_nodes.remove(node_in)

    return top_nodes


def is_nonterminal(node):
    return node.startswith('X')


def is_gpred(node):
    return not node.startswith('_') and '_card' not in node and '_ord' not in node


def is_noun(node):
    return get_node_pos(node) in {'n', 'np'} or node.startswith('pron') or node.startswith('person') or \
           '_named' in node or '_card' in node or '_ord' in node or '/NNS_' in node or '/NN_' in node or \
           '_yofc' in node or '_mofy' in node or '_dofm' in node or '_dofw' in node or '_year_range' in node or \
            '_season_' in node


def is_conj(node):
    return get_node_pos(node) == 'c' or 'subord' in node or node == 'appos' or node.startswith('implicit_conj')


def is_prep(node):
    return get_node_pos(node) == 'p' or node == 'loc_nonsp' or node == 'comp_equal' or \
           node == 'interval_p_start' or node == 'interval_p_end'


def is_verb(node):
    return get_node_pos(node) == 'v'


def is_basic_quantifier(nodes, edges):
    if not len(nodes) == 2 or not len(edges) == 1:
        return False

    if not get_node_pos(nodes[0]) == 'q':
        return False

    if not is_nonterminal(nodes[1]):
        return False

    return True


def is_basic_modifier(nodes, edges):
    if not len(nodes) == 2 or not len(edges) == 1:
        return False

    if not get_node_pos(nodes[0]) == 'a':
        return False

    if not is_nonterminal(nodes[1]):
        return False

    return True


def is_basic_noun(nodes, edges):
    if any([True if x.startswith('nominalization') else False for x in nodes]):
        if not len(nodes) == 2 or not len(edges) == 1:
            return False

        if is_nonterminal(nodes[1]):
            return False

    else:
        if not len(nodes) == 1 or not len(edges) == 0:
            return False

        if not is_noun(nodes[0]):
            return False

    return True


def is_basic_verb(nodes, edges):

    verb_index = None
    for index, node in enumerate(nodes):
        if is_verb(node) and verb_index is None:
            verb_index = str(index)

        elif not is_nonterminal(node):
            return False

    if verb_index is None:
        return False

    return True


def is_basic_conj(nodes, edges):
    if not len(nodes) >= 3 or not len(edges) >= 2:
        return False

    conj_index = None
    for index, node in enumerate(nodes):
        if is_conj(node) and conj_index is None:
            conj_index = str(index)

        elif not is_nonterminal(node):
            return False

    if conj_index is None:
        return False

    return True


def is_basic_prep(nodes, edges):
    if not len(nodes) >= 3 or not len(edges) >= 2:
        return False

    prep_index = None
    for index, node in enumerate(nodes):
        if is_prep(node) and prep_index is None:
            prep_index = str(index)

        elif not is_nonterminal(node):
            return False

    if prep_index is None:
        return False

    return True


def is_noun_compound(nodes, edges):

    # At least two children node of compound and 2 edges between them
    if not len(nodes) >= 3 or not len(edges) >= 2:
        return False

    # No top node can be c, p, v
    top_nodes = get_top_nodes(nodes, edges)

    for node in top_nodes:
        if is_verb(node) or is_conj(node) or is_prep(node):
            return False

    # compound must be a top node
    if 'compound' not in top_nodes:
        return False

    return True


def is_full_noun_phrase(nodes, edges):

    # At least 2 nodes and an edge between them
    if not len(nodes) >= 2 or not len(edges) >= 1:
        return False

    noun_indexes = []
    for index, node in enumerate(nodes):

        # Prevent any non-terminals
        if node.startswith('X'):
            return False

        # Collect noun nodes
        if is_noun(node) or node.startswith('nominalization'):
            noun_indexes.append(str(index))

    # At least one noun node needs to be present
    if len(noun_indexes) == 0:
        return False

    # No top node can be c, p, v
    top_nodes = get_top_nodes(nodes, edges)

    for index, node in enumerate(top_nodes):
        if is_verb(node) or is_conj(node) or is_prep(node):
            return False

    return True


def is_partial_noun_phrase(nodes, edges):

    # At least 2 nodes and an edge between them
    if not len(nodes) >= 2 or not len(edges) >= 1:
        return False

    # At least 1 nonterminal
    nonterminal_count = 0
    for index, node in enumerate(nodes):
        if is_nonterminal(node):
            nonterminal_count += 1

    if nonterminal_count == 0:
        return False

    # No top node can be c, p, v, gpred
    top_nodes = get_top_nodes(nodes, edges)

    for node in top_nodes:
        if is_verb(node) or is_conj(node) or is_prep(node):
            return False

        if is_gpred(node) and node != 'poss' and not node.startswith('nominalization'):
            return False

    return True


def is_verb_phrase(nodes, edges):

    # At least 2 nodes and an edge between them (verb and argument)
    if not len(nodes) >= 2 or not len(edges) >= 1:
        return False

    # No top node can be c, p
    top_nodes = get_top_nodes(nodes, edges)

    verb_top_count = 0
    for node in top_nodes:
        if is_conj(node) or is_prep(node):
            return False

        elif is_verb(node):
            verb_top_count += 1

    # At least one top node has to be v
    if verb_top_count == 0:
        return False

    return True


def is_mod_verb_phrase(nodes, edges):

    # No top node can be c, p
    top_nodes = get_top_nodes(nodes, edges)

    mods = []
    for node in top_nodes:
        if is_conj(node) or is_prep(node):
            return False

        if get_node_pos(node) == 'a' or node == 'neg':
            mods.append(node)

    mod_verbs = set()
    for edge in edges:
        node_out = nodes[int(edge[0])]
        node_in = nodes[int(edge[1])]

        if node_out in mods and is_verb(node_in):
            mod_verbs.add(node_in)

    # At least one modified verb needs to be present
    if len(mod_verbs) == 0:
        return False

    return True


def is_conj_phrase(nodes, edges):

    # At least 3 nodes and two edges between them (conj and 2 arguments)
    if not len(nodes) >= 3 or not len(edges) >= 2:
        return False

    # At least one top node is conj
    top_nodes = get_top_nodes(nodes, edges)

    conj_top_count = 0
    for node in top_nodes:
        if is_conj(node):
            conj_top_count += 1

    if conj_top_count == 0:
        return False

    return True


def is_prep_phrase(nodes, edges):

    # At least 3 nodes and two edges between them (prep and 2 arguments)
    if not len(nodes) >= 3 or not len(edges) >= 2:
        return False

    # At least one top node is prep
    top_nodes = get_top_nodes(nodes, edges)

    prep_top_count = 0
    for node in top_nodes:
        if is_prep(node):
            prep_top_count += 1

    if prep_top_count == 0:
        return False

    return True


def assign_rule_type(source_nodes, source_edges):

    if is_basic_quantifier(source_nodes, source_edges):
        return 'quant_basic'

    elif is_basic_modifier(source_nodes, source_edges):
        return 'mod_basic'

    elif is_basic_noun(source_nodes, source_edges):
        return 'noun_basic'

    elif is_basic_verb(source_nodes, source_edges):
        return 'verb_basic'

    elif is_basic_conj(source_nodes, source_edges):
        return 'conj_basic'

    elif is_basic_prep(source_nodes, source_edges):
        return 'prep_basic'

    elif is_verb_phrase(source_nodes, source_edges):
        return 'verb_phrase'

    elif is_mod_verb_phrase(source_nodes, source_edges):
        return 'verb_mod_phrase'

    elif is_noun_compound(source_nodes, source_edges):
        return 'noun_comp'

    elif is_full_noun_phrase(source_nodes, source_edges):
        return 'noun_full_phrase'

    elif is_partial_noun_phrase(source_nodes, source_edges):
        return 'noun_partial_phrase'

    elif is_conj_phrase(source_nodes, source_edges):
        return 'conj_phrase'

    elif is_prep_phrase(source_nodes, source_edges):
        return 'prep_phrase'

    else:
        return 'x_no_match'
