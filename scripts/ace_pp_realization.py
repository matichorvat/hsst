#!/usr/bin/env python

import re
import sys
import argparse
import subprocess

replacement_dict = {
    'NN': [('chocolate', 'chocolate', '_chocolate_n_1'), ('wood', 'wood', '_wood_n_1'), ('paper', 'paper', '_paper_n_1'), ('cherry', 'cherry', '_cherry_n_1'), ('dog', 'dog', '_dog_n_1')],
    'NNS': [('cat', 'cats', '_cat_n_1'), ('rose', 'roses', '_rose_n_1')],
    'VBD': [('shout', 'shouted', '_shout_v_1'), ('spell', 'spelled')],
    'RB': [('abstract', 'abstractly', '_abstract_a_1'), ('quick', 'quickly', '_quick_a_1')],
    'JJ': [('sweet', 'sweet', '_sweet_a_to'), ('yellow', 'yellow', '_yellow_a_1')],
    'JJR': [('sweet', 'sweeter', '_sweet_a_to'), ('clean', 'cleaner', '_clean_a_1')],
    'JJS': [('sweet', 'sweetest', '_sweet_a_to'), ('clean', 'cleanest', '_clean_a_1')],
    'VB': [('shout', 'shout', '_shout_v_1'), ('spell', 'spell', '_spell_v_1')],
    'VBN': [('shout', 'shouted', '_shout_v_1'), ('spell', 'spelled', '_spell_v_1')],
    'VBG': [('shout', 'shouting', '_shout_v_1'), ('sleep', 'sleeping', '_sleep_v_1')],
    'VBZ': [('shout', 'shouts', '_shout_v_1'), ('spell', 'sleeps', '_spell_v_1')],
    'FW':[('mais', 'mais', '_mais_n_1')]
}

predicate_regex = re.compile('_([^ <]+)_[avn]')
unknown_predicate_regex = re.compile('_([^ ]+)/([A-Z]{2,3})_u_unknown')


def preprocess_mrs(mrs):
    real_pred_matches = list(predicate_regex.finditer(mrs))
    unknown_pred_matches = list(unknown_predicate_regex.finditer(mrs))

    if len(unknown_pred_matches) == 0:
        raise Exception('No match found when there should be one.')

    sentence_pred_lemmas = {real_pred_match.group(1) for real_pred_match in real_pred_matches}

    # Replace unknown preds in MRS
    mrs, mappings = replace_unknown_preds(mrs, unknown_pred_matches, sentence_pred_lemmas)
    return mrs, mappings


def replace_unknown_preds(mrs, unknown_pred_matches, sentence_pred_lemmas):
    mappings = {}

    for unknown_pred_match in unknown_pred_matches:
        unknown_pred_string = unknown_pred_match.group(1)
        unknown_pred_pos_tag = unknown_pred_match.group(2)

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
    mappings = None

    if '_u_unknown' in mrs:
        mrs, mappings = preprocess_mrs(mrs)

    realizations = call_ace(ace_real_cmd, mrs.encode('utf-8'))

    if mappings is not None:
        realizations = postprocess_realizations(realizations, mappings)
        #print mappings
        #print realizations

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
