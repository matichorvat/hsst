import logging
import xml.etree.ElementTree as xml

from hsst.utility import timeout as to
from hsst.utility.graph import read_graphs
from hsst.utility.representation import loads
from hsst.ruleapplication.ruleapplication import ruleapplication


def string_dataset_ruleapplication(sentence_segment, format='xml', idx=True, id=-1, **kwargs):
    try:
        sentence_collection = loads(sentence_segment, format=format)
    except xml.ParseError:
        raise Exception('Error parsing sentence XML for id %d' % id)

    return dataset_ruleapplication(sentence_collection, idx=idx, **kwargs)


def dataset_ruleapplication(sentence_collection, idx=True, filtering=None, timeout=None, **kwargs):
    read_graphs(sentence_collection, idx=idx)

    graph_coverages = dict()

    for sentence in sentence_collection:
        sent_id, applied_rules = sentence_ruleapplication(sentence, filtering=filtering, timeout=timeout, **kwargs)
        graph_coverages[sent_id] = applied_rules

    return graph_coverages


def string_sentence_ruleapplication(sentence_segment, format='xml', idx=True, id=-1, filtering=None, timeout=None, **kwargs):
    try:
        sentence_collection = loads(sentence_segment, format=format)
    except xml.ParseError:
        raise Exception('Error parsing sentence XML for id %d' % id)

    read_graphs(sentence_collection, idx=idx)

    return sentence_ruleapplication(sentence_collection[0], filtering=filtering, timeout=timeout, **kwargs)


def sentence_ruleapplication(sentence, filtering=None, timeout=None, **kwargs):

    applied_rules = dict()

    if filtering is not None and not filtering.filter(sentence):
        logging.info('Skipping sentence with id %s (%s) due to filtering (graph size %d)' % (
            sentence.id, sentence.orig_id, len(sentence.source.graph)))
        return sentence.id, applied_rules

    if sentence.source.graph is not None:
        logging.debug('Starting rule application for sentence %s' % sentence.id)

        try:
            if timeout:
                with to.timeout(seconds=timeout):
                    applied_rules = ruleapplication(sentence.source.graph, **kwargs)
            else:
                applied_rules = ruleapplication(sentence.source.graph, **kwargs)

            logging.info('Extracted %d possible source sides for sentence with id %s (%s)' % (
                len(applied_rules), sentence.id, sentence.orig_id))

        except to.TimeoutError:
            logging.warn('Rule application for sentence with id %s (%s) failed due to timeout after %d seconds' % (
                sentence.id, sentence.orig_id, timeout))

    return sentence.id, applied_rules
