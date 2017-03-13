import logging
from collections import defaultdict


def align(src_side, trg_side, token_index_alignment):

    sttt = None
    sgtt = None
    sttg = None
    sgtg = None

    if src_side.tokenized_text is not None and trg_side.tokenized_text is not None:
        sttt = align_sttt(src_side.tokenized_text, trg_side.tokenized_text, token_index_alignment)

    if src_side.dmrs is not None and trg_side.tokenized_text is not None:
        if token_index_alignment is not None:
            sgtt = align_sgtt_trans(src_side.dmrs, trg_side.tokenized_text, token_index_alignment)
        else:
            sgtt = align_sgtt_real(src_side.dmrs, trg_side.tokenized_text)

        # print xml.tostring(src_side.dmrs, encoding='utf-8')
        # print ' '.join(src_side.tokenized_text)
        # for tok_align, ind_align in sgtt:
        #     print ind_align, tok_align
        #
        # print '*'*100

    if src_side.tokenized_text is not None and trg_side.dmrs is not None:
        sttg = align_sttg(src_side.tokenized_text, trg_side.dmrs, token_index_alignment)

    if src_side.dmrs is not None and trg_side.dmrs is not None:
        sgtg = align_sgtg(src_side.dmrs, trg_side.dmrs, token_index_alignment)

    return sttt, sgtt, sttg, sgtg


def align_sttt(src_tok, trg_tok, tok_index_align):

    try:
        tok_align = [(src_tok[src_index], trg_tok[trg_index]) for src_index, trg_index in tok_index_align]
        return zip(tok_align, tok_index_align)

    except IndexError:
        logging.exception(src_tok)
        logging.exception(trg_tok)
        logging.exception(tok_index_align)
        logging.exception('Non-existent token index: %s' % tok_index_align)
        raise Exception('Non-existent token index: %s' % tok_index_align)


def align_sgtt_trans(src_dmrs, trg_tok, tok_index_align):
    src_align_dict = defaultdict(list)
    sgtt_align = list()

    for src_index, trg_index in tok_index_align:
        src_align_dict[src_index].append(trg_index)

    for entity in src_dmrs:
        if entity.tag == 'node':
            node = entity
            node_id = node.attrib['nodeid']
            node_label = node.attrib['label']
            node_src_tok_indexes = [int(x) for x in node.attrib['tokalign'].split(' ') if x.strip() != '']

            node_trg_tok_indexes = list()
            for src_tok_index in node_src_tok_indexes:
                node_trg_tok_indexes.extend(src_align_dict[src_tok_index])

            node_trg_tok_indexes = sorted(list(set(node_trg_tok_indexes)))
            node_trg_toks = [trg_tok[trg_index] for trg_index in node_trg_tok_indexes]

            sgtt_align.append(((node_label, node_trg_toks), (node_id, node_trg_tok_indexes)))

    return sgtt_align


def align_sgtt_real(src_dmrs, trg_tok):
    sgtt_align = list()

    for entity in src_dmrs:
        if entity.tag == 'node':
            node = entity
            node_id = node.attrib['nodeid']
            node_label = node.attrib['label']

            node_trg_tok_indexes = [int(x) for x in node.attrib['tokalign'].split(' ') if x.strip() != '' and x != '-1']
            node_trg_tok_indexes = sorted(list(set(node_trg_tok_indexes)))
            node_trg_toks = [trg_tok[trg_index] for trg_index in node_trg_tok_indexes]

            sgtt_align.append(((node_label, node_trg_toks), (node_id, node_trg_tok_indexes)))

    return sgtt_align


def align_sttg(src_tok, trg_dmrs, tok_align):
    raise NotImplementedError('Parse sttg alignment not supported yet.')


def align_sgtg(src_dmrs, trg_dmrs, tok_align):
    raise NotImplementedError('Parse sttg alignment not supported yet.')
