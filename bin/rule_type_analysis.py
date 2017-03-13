import ast
import random
from collections import defaultdict

from hsst.rulextraction.rule_type import load_wmap, assign_rule_type


def parse_rule(rule_string, dmrs_wmap, de_wmap):
    source_side = rule_string.split('\t')[1]
    target_side = rule_string.split('\t')[2]
    features = ast.literal_eval(rule_string.split('\t')[3])

    nodes, edges = source_side.split('|')
    nodes = [dmrs_wmap[node_id] if node_id not in {'X0', 'X1'} else node_id for node_id in nodes.split('_')]
    edges = [edge.split('-') for edge in edges.split('_') if edge != '']
    for edge in edges:
        edge[2] = dmrs_wmap[edge[2]]

    target = ' '.join([de_wmap[idx] if not idx.startswith('X') else idx for idx in target_side.split('_')])

    return nodes, edges, target, features


def analyze(rule_types, show_all=False, sample=0):
    total_sum = sum(len(rule_types[x]) for x in rule_types)

    print '*' * 50
    print 'Grammar partition'
    print '*' * 50

    for rule_type, rules in sorted(rule_types.items()):
        print rule_type, '{0:.1f}%'.format(len(rules) / float(total_sum) * 100)

    print '*' * 50

    if show_all or sample > 0:
        for rule_type, rules in sorted(rule_types.items()):
            print rule_type, '{0:.1f}%'.format(len(rules) / float(total_sum) * 100)

            if sample > 0:
                rules = random.sample(rules, sample) if sample < len(rules) else rules

            print '*' * 50
            for rule in rules:
                print rule[0], rule[1], rule[2].encode('utf-8')
            print '*' * 50


if __name__ == '__main__':

    random.seed(1)

    directory = '/data/mifs_scratch/mh693/wmt15/rule_types/100kNOF/'

    dmrs_wmap = load_wmap(directory + 'dmrs.wmap')
    de_wmap = load_wmap(directory + '/de.wmap')

    rule_types = defaultdict(list)

    with open(directory + '/grammar.txt') as grammar:

        for rule in grammar:
            source_nodes, source_edges, target, _ = parse_rule(rule, dmrs_wmap, de_wmap)

            rule_type = assign_rule_type(source_nodes, source_edges)
            rule_types[rule_type].append((source_nodes, source_edges, target))

    analyze(rule_types, sample=50)
