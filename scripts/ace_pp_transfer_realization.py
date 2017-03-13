#!/usr/bin/env python

import re
import sys
import argparse
import subprocess

replacement_dict = {
    'n': [
        ('chocolate', 'chocolate', '_chocolate_n_1_rel'),
        ('wood', 'wood', '_wood_n_1_rel'),
        ('paper', 'paper', '_paper_n_1_rel'),
        ('cherry', 'cherry', '_cherry_n_1_rel'),
        ('dog', 'dog', '_dog_n_1_rel')
    ],
    'a': [
        ('sweet', 'sweet', '_sweet_a_to_rel'),
        ('yellow', 'yellow', '_yellow_a_1_rel')
    ],
    'v': [
        ('shout', 'shout', '_shout_v_1_rel'),
        ('spell', 'spell', '_spell_v_1_rel'),
        ('write', 'write', '_write_v_1_rel')
    ],
    'p': [
        ('for', 'for', '_for_p_rel'),
        ('from', 'from', '_from_p_rel')
    ],
    'c': [
        ('and', 'and', '_and_c_rel'),
        ('so', 'so', '_so_c_rel')
    ]
}

gpred_map = {
    'discourse_x_rel': 'discourse_rel',
    'ellipses_rel': 'ellipsis_rel',
    'if_then_rel': 'if_x_then_rel',
    'neg_x_rel': 'neg_rel',
    'number_rel': 'number_q_rel',
    'part-of_rel': 'part_of_rel',
    'unknown_v_rel': 'unknown_rel',
    'unknown_v_cop_rel': 'cop_id_rel',
    'unspec_p_manner_rel': 'unspec_manner_rel'
}

gpred_drop = {
    'about_rel_noun_mark',
    'for_rel_noun_mark',
    'of_rel_noun_mark',
    'poss_mark',
    'to_rel_noun_mark',
    'adj_from_noun_mark'
}

predicate_regex = re.compile('_([^ <]+)_[avn]')
unknown_predicate_regex = re.compile('"ja:_([^ ]+_rel)"')
unknown_gpredicate_regex = re.compile('"ja:([^ ]+_rel)"')
drop_gpred_regex = re.compile('\[ "([^ ]+_mark)"[^\]]+ \]')


def preprocess_mrs(mrs):
    # Find all preds in MRS
    real_pred_matches = list(predicate_regex.finditer(mrs))
    sentence_pred_lemmas = {real_pred_match.group(1) for real_pred_match in real_pred_matches}

    # Find and replace unknown real preds in MRS
    unknown_pred_matches = list(unknown_predicate_regex.finditer(mrs))
    mrs, mappings = replace_unknown_preds(mrs, unknown_pred_matches, sentence_pred_lemmas)

    # Find and replace unknown grammar preds in MRS
    unknown_gpred_matches = list(unknown_gpredicate_regex.finditer(mrs))
    mrs = preprocess_unknown_gpreds(mrs, unknown_gpred_matches)

    # Find and replace drop grammar preds in MRS
    drop_gpred_matches = list(drop_gpred_regex.finditer(mrs))
    mrs = preprocess_drop_gpreds(mrs, drop_gpred_matches)

    return mrs, mappings


def replace_unknown_preds(mrs, unknown_pred_matches, sentence_pred_lemmas):
    mappings = {}

    for unknown_pred_match in unknown_pred_matches:
        unknown_pred_label = unknown_pred_match.group(1)

        unknown_pred_label_parts = unknown_pred_label.split('_')
        unknown_pred_string = unknown_pred_label_parts[0]
        unknown_pred_pos_tag = unknown_pred_label_parts[1]

        if unknown_pred_pos_tag not in replacement_dict:
            # sys.stderr.write('Type {} not in replacement dictionary'.format(unknown_pred_pos_tag))
            continue

        replacement_pred = None
        for unknown_pred_replacement in replacement_dict[unknown_pred_pos_tag]:
            if unknown_pred_replacement[0] not in sentence_pred_lemmas and unknown_pred_replacement[1] not in mappings:
                replacement_pred = unknown_pred_replacement
                break

        if replacement_pred is None:
            raise Exception('No replacement could be found for %s' % (unknown_pred_match.group(0),))

        mrs = re.sub(unknown_pred_match.group(0), replacement_pred[2], mrs)
        mappings[replacement_pred[1]] = unknown_pred_string

    return mrs, mappings


def preprocess_unknown_gpreds(mrs, unknown_gpred_matches):
    for unknown_pred_match in unknown_gpred_matches:
        gpred = unknown_pred_match.group(1)

        if gpred in gpred_map:
            gpred = gpred_map[gpred]

        mrs = re.sub(unknown_pred_match.group(0), gpred, mrs)

    return mrs


def preprocess_drop_gpreds(mrs, drop_gpred_matches):
    for drop_gpred_match in drop_gpred_matches:
        gpred = drop_gpred_match.group(1)

        if gpred in gpred_drop:
            mrs = mrs.replace(drop_gpred_match.group(0), '')

    return mrs


def postprocess_realizations(realizations, mappings):
    return [postprocess_realization(realization, mappings) for realization in realizations]


def postprocess_realization(realization, mappings):
    for replacement, original in mappings.items():
        realization = re.sub(replacement, original, realization)
    return realization


def call_ace(ace_real_cmd, mrs):
    p = subprocess.Popen(ace_real_cmd, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    stdout, stderr = p.communicate(input=mrs)
    realizations = [line.strip().decode('utf-8') for line in stdout.split('\n') if line.strip() != '']

    sys.stderr.write(stderr)
    #if len(realizations) == 0:
    #    print stderr

    return realizations


def realize(ace_real_cmd, mrs):
    mrs, mappings = preprocess_mrs(mrs)
    realizations = call_ace(ace_real_cmd, mrs.encode('utf-8'))

    if len(mappings) > 0:
        realizations = postprocess_realizations(realizations, mappings)

    return realizations


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run ACE parser with pre- and post-processing for unknown words.')
    parser.add_argument('ace_real_cmd')

    args = parser.parse_args()

    while True:
        try:
            mrs = raw_input().decode('utf-8').strip()
            realizations = realize(args.ace_real_cmd, mrs)
            sys.stdout.write('%s\n' % '\n'.join(x.encode('utf-8') for x in realizations))

        except EOFError:
            break
