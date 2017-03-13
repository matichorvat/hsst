import sys
import logging
import argparse

from hsst.preprocessing.filtering import dataset_filter, SourceGraphSizeFilter
from hsst.utility.representation import load, dump, dumps


def relabel_ids(sents):

    original_ids = []

    for index, sent in enumerate(sents, 1):
        original_ids.append(sent.id)
        sent.id = index

    return original_ids


if __name__ == "__main__":

    appName = 'Filtering'

    parser = argparse.ArgumentParser(description='Filter DS file by graph size.')
    parser.add_argument('-l', '--log')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-fmax', '--filtermax', type=int, default=20)
    parser.add_argument('-fmin', '--filtermin', type=int, default=0)
    parser.add_argument('--id_out')
    parser.add_argument('--orig_id_out')
    parser.add_argument('--target_out')
    parser.add_argument('--target_untok_out')
    parser.add_argument('--target_idx_out')
    parser.add_argument('--source_out')
    parser.add_argument('--source_untok_out')
    parser.add_argument('--source_idx_out')
    parser.add_argument('dataset_filepath')
    parser.add_argument('output_filepath')

    args = parser.parse_args()

    # Set up logging
    if args.log:
        logging.basicConfig(filename=args.log, level=logging.DEBUG if args.verbose else logging.INFO)
    else:
        logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    filtering = SourceGraphSizeFilter(max_nodes=args.filtermax, min_nodes=args.filtermin)

    with open(args.dataset_filepath) as fin, open(args.output_filepath, 'wb') as fout:

        print 'Loading sentences from', args.dataset_filepath
        sents = load(fin)

        print 'Filtering sentences'
        filtered_sents = dataset_filter(sents, filtering)

        print 'Number of sentences after filtering:', len(filtered_sents)

        print 'Relabeling sentence ids'
        original_ids = relabel_ids(filtered_sents)

        if args.id_out:
            print 'Saving original sentence ids to', args.id_out
            with open(args.id_out, 'wb') as fidout:
                fidout.write('\n'.join(original_ids))

        print 'Saving filtered sentence to', args.output_filepath
        dump(filtered_sents, fout, encode=False)

        if args.orig_id_out:
            print 'Outputting orig sids to', args.orig_id_out
            with open(args.orig_id_out, 'wb') as out:
                sids = dumps(filtered_sents, format='text', src=False, trg=False, align=False, sid=True)
                orig_sids = [sid.split()[2] for sid in sids.split('\n') if sid.strip() != '']
                out.write('\n'.join(orig_sids) + '\n')

        if args.target_out:
            print 'Outputting target tok to', args.target_out
            with open(args.target_out, 'wb') as out:
                dump(filtered_sents, out, format='text', src=False, align=False, plain=False, idx=False, dmrs=False)

        if args.target_untok_out:
            print 'Outputting target untok to', args.target_untok_out
            with open(args.target_untok_out, 'wb') as out:
                dump(filtered_sents, out, format='text', src=False, align=False, tok=False, idx=False, dmrs=False)

        if args.target_idx_out:
            print 'Outputting target idx to', args.target_idx_out
            with open(args.target_idx_out, 'wb') as out:
                dump(filtered_sents, out, format='text', src=False, align=False, plain=False, tok=False, dmrs=False)

        if args.source_out:
            print 'Outputting source tok to', args.source_out
            with open(args.source_out, 'wb') as out:
                dump(filtered_sents, out, format='text', trg=False, align=False, plain=False, idx=False, dmrs=False)

        if args.source_untok_out:
            print 'Outputting source untok to', args.source_untok_out
            with open(args.source_untok_out, 'wb') as out:
                dump(filtered_sents, out, format='text', trg=False, align=False, tok=False, idx=False, dmrs=False)

        if args.source_idx_out:
            print 'Outputting source idx to', args.source_idx_out
            with open(args.source_idx_out, 'wb') as out:
                dump(filtered_sents, out, format='text', trg=False, align=False, plain=False, tok=False, dmrs=False)
