
opening_punctuation = {'(': ')', '[': ']', '{': '}'}
closing_punctuation = {')': '(', ']': '[', '}': '{'}
pair_punctuation = {'"'}
mid_punctuation = {',', "'", ':', ';'}
end_punctuation = {'!', '?', ';'}

generation_ratio_limits = {1: (1, 3), 2: (1, 5), 3: (1, 7), 4: (1, 9), 5: (2, 11)}


def undergeneration_filter(rule, ratio=generation_ratio_limits):
    source_nodes, source_edges, target = rule

    if len(target.split(' ')) >= ratio[len(source_nodes)][0]:
        return True

    #if features['rule_type'] == 'noun_comp':
    #    return True

    return False


def overgeneration_filter(rule, ratio=generation_ratio_limits):
    source_nodes, source_edges, target = rule

    if len(target.split(' ')) <= ratio[len(source_nodes)][1]:
        return True

    return False


def url_filter(rule):
    source_nodes, source_edges, target = rule
    target_tokens = target.split(' ')

    for token in target_tokens:
        if token.startswith('www.') or token.endswith('.com') or token.endswith('.net') or token.endswith('.org') \
                or token.endswith('.ru') or token.endswith('.de') or token.endswith('.uk') or token.endswith('.jp'):
            return False

    return True


def unclosed_punctuation_filter(rule, opening_punctuation=opening_punctuation, closing_punctuation=closing_punctuation,
                                pair_punctuation=pair_punctuation):
    source_nodes, source_edges, target = rule
    target_tokens = target.split(' ')

    expected_closing = []
    for token in target_tokens:
        if token in opening_punctuation:
            expected_closing.append(opening_punctuation[token])

        elif token in closing_punctuation:
            if len(expected_closing) == 0:
                return False

            elif not token == expected_closing[-1]:
                return False

            elif token == expected_closing[-1]:
                expected_closing.pop()

        elif token in pair_punctuation:
            if len(expected_closing) > 0 and token == expected_closing[-1]:
                expected_closing.pop()

            else:
                expected_closing.append(token)

    if len(expected_closing) > 0:
        return False

    return True


def mid_end_punctuation_filter(rule, mid_punctuation=mid_punctuation, end_punctuation=end_punctuation):
    source_nodes, source_edges, target = rule
    target_tokens = target.split(' ')

    if target_tokens[0] in mid_punctuation or target_tokens[-1] in mid_punctuation:
        return False

    for token in target_tokens[1:-1]:
        if token in end_punctuation:
            return False

    return True


def apply_filters(rule):

    if not undergeneration_filter(rule):
        return 'undergeneration_filter'

    elif not overgeneration_filter(rule):
        return 'overgeneration_filter'

    elif not url_filter(rule):
        return 'url_filter'

    elif not unclosed_punctuation_filter(rule):
        return 'unclosed_punctuation_filter'

    elif not mid_end_punctuation_filter(rule):
        return 'mid_end_punctuation_filter'

    else:
        return 'x_no_filter'
