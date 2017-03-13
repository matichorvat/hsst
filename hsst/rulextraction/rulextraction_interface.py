import logging
import xml.etree.ElementTree as xml

from hsst.utility import timeout as to
from hsst.utility.graph import read_graphs
from hsst.utility import representation, utility
from hsst.rulextraction.rulextraction import rulextraction


def string_dataset_translation_rulextraction(sentence_segment, format='xml', idx=True, id=None, **kwargs):
    try:
        sentence_collection = representation.loads(sentence_segment, format=format)
    except xml.ParseError:
        raise Exception('Error parsing sentence XML for id %d' % id)

    return dataset_translation_rulextraction(sentence_collection, idx=idx, **kwargs)


def dataset_translation_rulextraction(sentence_collection, idx=True, filtering=None, timeout=None, **kwargs):
    read_graphs(sentence_collection, idx=idx)

    rules = list()

    for sentence in sentence_collection:
        if idx:
            tok = sentence.target.tokenized_idx
        else:
            tok = sentence.target.tokenized_text

        if filtering is not None and not filtering.filter(sentence):
            logging.info('Skipping sentence with id %s (%s) due to filtering (source token number %d, graph size %d)' % (
                sentence.id, sentence.orig_id, len(sentence.source.tokenized_text), len(sentence.source.graph)))
            continue

        if sentence.source.graph is not None and tok is not None and sentence.alignment is not None and sentence.alignment.sgtt is not None:
            logging.debug('Starting rule extraction for sentence %s' % sentence.id)

            alignments = dict(index for plain, index in sentence.alignment.sgtt)
            alignment_dict = utility.create_alignment_dict(alignments, sentence.source.graph)

            tok = [unicode(x) for x in tok]

            try:
                if timeout:
                    with to.timeout(seconds=timeout):
                        sentence_rules = rulextraction(sentence.source.graph, tok, alignment_dict, **kwargs)
                else:
                    sentence_rules = rulextraction(sentence.source.graph, tok, alignment_dict, **kwargs)

                rules.extend(sentence_rules)
                logging.info('Extracted %d rules from sentence with id %s (%s)' % (
                    len(sentence_rules), sentence.id, sentence.orig_id))

            except to.TimeoutError:
                logging.warn('Rule extraction for sentence with id %s (%s) failed due to timeout after %d seconds' % (
                    sentence.id, sentence.orig_id, timeout))
                continue

    return rules


def string_dataset_realization_rulextraction(sentence_segment, format='xml', idx=True, **kwargs):
    sentence_collection = representation.loads(sentence_segment, format=format)
    return dataset_realization_rulextraction(sentence_collection, idx=idx, **kwargs)


def dataset_realization_rulextraction(sentence_collection, idx=True, filtering=None, timeout=None, **kwargs):
    read_graphs(sentence_collection, idx=idx)

    rules = list()

    for sentence in sentence_collection:
        if idx:
            tok = sentence.source.tokenized_idx
        else:
            tok = sentence.source.tokenized_text

        if filtering is not None and not filtering.filter(sentence):
            logging.info('Skipping sentence with id %s (%s) due to filtering (token number %d, graph size %d)' % (
                sentence.id, sentence.orig_id, len(tok), len(sentence.source.graph)))
            continue

        if sentence.source.graph is not None and tok is not None:
            logging.debug('Starting rule extraction for sentence %s' % sentence.id)

            tok = [unicode(x) for x in tok]
            alignments = utility.create_realization_alignment(sentence.source.dmrs)
            alignment_dict = utility.create_alignment_dict(alignments, sentence.source.graph)

            try:
                if timeout:
                    with to.timeout(seconds=timeout):
                        sentence_rules = rulextraction(sentence.source.graph, tok, alignment_dict, **kwargs)
                else:
                    sentence_rules = rulextraction(sentence.source.graph, tok, alignment_dict, **kwargs)

                rules.extend(sentence_rules)
                logging.info('Extracted %d rules from sentence with id %s (%s)' % (
                    len(sentence_rules),
                    sentence.id,
                    sentence.orig_id
                ))

            except to.TimeoutError:
                logging.warn('Rule extraction for sentence with id %s (%s) failed due to timeout after %d seconds' % (
                    sentence.id,
                    sentence.orig_id,
                    timeout
                ))

                continue

    return rules
