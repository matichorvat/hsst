#!/usr/bin/env python

import os
import re
import shutil
import string
import argparse
import functools
import itertools


def score(hssr_input, gold_dir, preprocess_func, nbest, ignore_missing=False, hssr_nbest=0, csv=False):

    assert nbest >= 0 and hssr_nbest >= 0

    if nbest == 0:
        gold_sentences = read_1best(gold_dir, preprocess_func)
    else:
        gold_sentences = read_nbest(gold_dir, preprocess_func, nbest)

    if hssr_nbest == 0:
        hssr_sentences = read_1best(hssr_input, preprocess_func)
    else:
        hssr_sentences = read_nbest(hssr_input, preprocess_func, hssr_nbest)

    sentence_num = max(max(gold_sentences.keys()), max(hssr_sentences.keys()))

    sentence_matches = []
    unmatched_sentence_indexes = []
    for sentence_index in range(1, sentence_num + 1):

        gold_sentence_nbest = gold_sentences[sentence_index]
        hssr_sentence_nbest = hssr_sentences[sentence_index]

        matched = False
        for hssr_sentence in hssr_sentence_nbest:
            if hssr_sentence in gold_sentence_nbest:
                matched = True
                break

        if matched:
            sentence_matches.append(1)
        elif len(gold_sentence_nbest) > 0 or not ignore_missing:
            sentence_matches.append(0)
            unmatched_sentence_indexes.append(sentence_index)
            # print sentence, gold_sentence_outputs

    if not csv:
        print '# Accuracy score for HSSR {4}-best, ERG {3}-best is {0:.3f} ({1}/{2})'.format(
            sum(sentence_matches) / float(len(sentence_matches)) if len(sentence_matches) > 0 else 0,
            sum(sentence_matches),
            len(sentence_matches),
            nbest if nbest != 0 else 1,
            hssr_nbest if hssr_nbest != 0 else 1
        )
    else:
        print '{4},{3},{0:.3f},{1},{2}'.format(
            sum(sentence_matches) / float(len(sentence_matches)) if len(sentence_matches) > 0 else 0,
            sum(sentence_matches),
            len(sentence_matches),
            nbest,
            hssr_nbest if hssr_nbest != 0 else 1
        )


    return unmatched_sentence_indexes


def read_1best(filename, preprocess_func):
    sentences = {}
    with open(filename) as fde:
        for sentence_index, line in enumerate(fde, 1):
            sentence = preprocess_func(line)
            sentences[sentence_index] = [sentence]

    return sentences


def read_nbest(dirname, preprocess_func, nbest):
    file_map = {}
    for filename in os.listdir(dirname):
        file_map[int(os.path.splitext(os.path.basename(filename))[0])] = os.path.join(dirname, filename)

    sentences = {}
    for sentence_index in range(1, max(file_map.keys()) + 1):
        with open(file_map[sentence_index]) as fg:
            sentence_nbest = [preprocess_func(line) for line in fg if line.strip() != ''][:nbest]
            sentences[sentence_index] = sentence_nbest

    return sentences


def preprocess(line, whitespace_normalize=True, case_insensitive=False, end_punctuation=False, all_punctuation=False):
    line = line.strip()

    if case_insensitive:
        line = line.lower()

    if not all_punctuation and end_punctuation and len(line) > 0 and line[-1] in '.,!?':
        line = line[:-1]

    if all_punctuation:
        line = line.translate(string.maketrans('', ''), string.punctuation)

    if whitespace_normalize:
        line = re.sub('\s+', ' ', line).strip()

    return line


def output_unmatched(unmatched_sentence_indexes, input_dir, output_dir):
    try:
        shutil.rmtree(output_dir)
    except OSError:
        pass

    os.makedirs(output_dir)

    for sentence_index in unmatched_sentence_indexes:
        input_filepath = os.path.join(input_dir, str(sentence_index) + '.txt')
        shutil.copy(input_filepath, output_dir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Exact match scoring for realization outputs against ERG realizer n-best.')
    parser.add_argument('-w', '--whitespace_normalize', action='store_false', help='Do NOT normalize whitespace matching.')
    parser.add_argument('-c', '--case_insensitive', action='store_true', help='Use case insensitive matching.')
    parser.add_argument('-e', '--end_punctuation', action='store_true', help='Ignore end punctuation for matching.')
    parser.add_argument('-p', '--all_punctuation', action='store_true', help='Ignore all punctuation for matching.')
    parser.add_argument('-i', '--ignore_missing', action='store_true', help='Do not penalize score if no gold output available.')
    parser.add_argument('-n', '--nbest', default=5, help='Number of n-best erg realizations to consider. Multiple values can separated by commas.')
    parser.add_argument('--hssr_nbest', default=0, help='Number of n-best HSSR realizations to consider. Multiple values can separated by commas.')
    parser.add_argument('--output_unmatched', help='Specify directory where unmatched files are output')
    parser.add_argument('hssr_input')
    parser.add_argument('erg_dir')

    args = parser.parse_args()

    preprocess_func = functools.partial(
        preprocess,
        whitespace_normalize=args.whitespace_normalize,
        case_insensitive=args.case_insensitive,
        end_punctuation=args.end_punctuation,
        all_punctuation=args.all_punctuation
    )

    hssr_nbests = [0] if args.hssr_nbest == 0 else [int(x) for x in args.hssr_nbest.split(',')]
    erg_nbests = [int(x) for x in args.nbest.split(',')]

    print 'Evaluating cartesian product of n-bests', hssr_nbests, 'x', erg_nbests

    for hssr_nbest, erg_nbest in itertools.product(hssr_nbests, erg_nbests):
        unmatched_sentence_indexes = score(
            args.hssr_input,
            args.erg_dir,
            preprocess_func,
            int(erg_nbest),
            ignore_missing=args.ignore_missing,
            hssr_nbest=hssr_nbest
        )

        if args.output_unmatched:
            dirpath = os.path.join(args.output_unmatched, str(hssr_nbest) + '_' + str(erg_nbest))
            output_unmatched(unmatched_sentence_indexes, args.hssr_input, dirpath)
