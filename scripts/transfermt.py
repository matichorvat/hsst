#!/usr/bin/env python

import sys
import argparse
import subprocess


def translate(ace_transfermt_cmd, mrs):
    p = subprocess.Popen(ace_transfermt_cmd, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    stdout, stderr = p.communicate(input=mrs.encode('utf-8'))
    sys.stderr.write(stderr)
    translations = [line.strip().decode('utf-8') for line in stdout.split('\n') if line.strip() != '']

    return translations


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run ACE transfermt with post-processing.')
    parser.add_argument('--single_input', action='store_true', help='All input is for a single sentence')
    parser.add_argument('--n_best', '-n', default=0, type=int, help='Number of n-best transferred outputs to keep. 0 keeps all.')
    parser.add_argument('ace_transfermt_cmd')

    args = parser.parse_args()

    if args.single_input:
        sentence_translations = []

    num_inputs = 0
    while True:
        try:
            mrs = raw_input().decode('utf-8').strip()

            if mrs == '' or mrs.startswith('SENT:'):
                continue

            # Keep only MRS
            mrs = mrs.split(' ; ')[0]

            num_inputs += 1

            translations = translate(args.ace_transfermt_cmd, mrs)

            if args.single_input:
                if args.n_best == 0:
                    sentence_translations.extend(translations)
                else:
                    sentence_translations.extend(translations[0:args.n_best])

            else:
                if len(translations) == 0:
                    sys.stdout.write('\n')
                    continue

                if args.n_best == 0:
                    sys.stdout.write('%s\n' % '\n'.join(x.encode('utf-8') for x in translations))
                else:
                    sys.stdout.write('%s\n' % '\n'.join(x.encode('utf-8') for x in translations[0:args.n_best]))

        except EOFError:
            break

    sys.stderr.write('%d,%d\n' % (num_inputs, len(sentence_translations)))

    if args.single_input:
        if len(sentence_translations) == 0:
            sys.stdout.write('\n')
        else:
            sys.stdout.write('%s\n' % '\n'.join(x.encode('utf-8') for x in sentence_translations))
