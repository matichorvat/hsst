import random
from collections import defaultdict

from rule_type_analysis import parse_rule
from hsst.rulextraction.rule_type import load_wmap, assign_rule_type


def analyze(filter_types, show_all=False, sample=0):
    total_sum = sum(len(filter_types[x]) for x in filter_types)

    print '*' * 50
    print 'Filter partition'
    print '*' * 50

    for filter_type, rules in sorted(filter_types.items()):
        print filter_type, '{0:.1f}%'.format(len(rules) / float(total_sum) * 100)

    print '*' * 50

    if show_all or sample > 0:
        for filter_type, rules in sorted(filter_types.items()):
            print filter_type, '{0:.1f}%'.format(len(rules) / float(total_sum) * 100)

            if sample > 0:
                rules = random.sample(rules, sample) if sample < len(rules) else rules

            print '*' * 50
            for rule in rules:
                print rule[0], rule[1], rule[2].encode('utf-8')  # , rule[3], rule[4]
            print '*' * 50


if __name__ == '__main__':

    random.seed(1)

    directory = '/data/mifs_scratch/mh693/wmt15/rule_types/100k/'

    dmrs_wmap = load_wmap(directory + 'dmrs.wmap')
    de_wmap = load_wmap(directory + '/de.wmap')

    filter_types = defaultdict(list)

    with open(directory + '/grammar.txt') as grammar:

        for rule in grammar:
            rule = parse_rule(rule, dmrs_wmap, de_wmap)
            source_nodes, source_edges, target, features = rule
            rule = source_nodes, source_edges, target

            rule_type = assign_rule_type(source_nodes, source_edges)
            features['rule_type'] = rule_type

            filter_type = apply_filters(rule)

            filter_types[filter_type].append((source_nodes, source_edges, target, features, rule_type))

    analyze(filter_types, sample=50)
