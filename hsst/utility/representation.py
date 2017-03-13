import sys
import logging
import xml.etree.ElementTree as xml
from collections import defaultdict


class SentenceCollection(object):

    def __init__(self, sentences=None):
        if sentences is None:
            self.sentences = list()
        else:
            self.sentences = sentences

    def add(self, sentence):
        self.sentences.append(sentence)

    def find(self, sid):
        for sentence in self.sentences:
            if sentence.id == sid:
                return sentence

        return None

    def __iter__(self):
        return iter(self.sentences)

    def __getitem__(self, index):
        return self.sentences[index]

    def __len__(self):
        return len(self.sentences)


class Sentence(object):

    def __init__(self, sid, src_side=None, trg_side=None, alignment=None, result=None, orig_id=None):
        self.id = sid
        self.orig_id = orig_id
        self.source = src_side
        self.target = trg_side
        self.alignment = alignment
        self.result = result


class Side(object):

    def __init__(self, plain=None, tok=None, idx=None, dmrs=None):
        # string
        self.plain_text = plain
        # list
        self.tokenized_text = tok
        # list
        self.tokenized_idx = idx
        # xml
        self.dmrs = dmrs
        # Semantic Graph
        self.graph = None


class Alignment(object):

    def __init__(self, sttt=None, sgtt=None, sttg=None, sgtg=None):
        # Nested tuples
        self.sttt = sttt
        self.sgtt = sgtt
        self.sttg = sttg
        self.sgtg = sgtg


class Result(object):

    def __init__(self, n_best_fst=None, n_best_list=None):
        # FST
        self.n_best_fst = n_best_fst
        # [(idx, plain, score)]
        self.n_best_list = n_best_list


def dump(sent_col, fp, format='xml', src=True, trg=True, align=True, plain=True, tok=True, idx=True, dmrs=True, encode=True, subset=None, sid=True):
    dump_string = dumps(sent_col, format, src, trg, align, plain, tok, idx, dmrs, subset, sid)

    if encode:
        fp.write(dump_string.encode('utf-8'))
    else:
        fp.write(dump_string)


def dumps(sent_col, format='xml', src=True, trg=True, align=True, plain=True, tok=True, idx=True, dmrs=True, subset=None, sid=True):
    if format == 'xml':
        sentence_space = '\n\n'
    else:
        sentence_space = '\n'

    sent_str_list = list()
    for sentence in sent_col:
        if subset is not None and sentence.id in subset:
            sent_str_list.append(sentence_dumps(sentence, format, src, trg, align, plain, tok, idx, dmrs, sid))
        elif subset is None:
            sent_str_list.append(sentence_dumps(sentence, format, src, trg, align, plain, tok, idx, dmrs, sid))

    return sentence_space.join(sent_str_list) + '\n'


def sentence_dumps(sentence, format='xml', src=True, trg=True, align=True, plain=True, tok=True, idx=True, dmrs=True, sid=True):
    if format == 'xml':
        snt_root = xml.Element('sentence', attrib={'id': str(sentence.id)})

        if sentence.orig_id is not None:
            snt_root.attrib['orig_id'] = sentence.orig_id

        snt_root.text = '\n'

        if src and sentence.source is not None:
            src_root = side_dumps(sentence.source, 'source', format, plain, tok, idx, dmrs)
            snt_root.append(src_root)

        if trg and sentence.target is not None:
            trg_root = side_dumps(sentence.target, 'target', format, plain, tok, idx, dmrs)
            snt_root.append(trg_root)

        if align and sentence.alignment is not None:
            align_root = align_dumps(sentence.alignment, format, tok, idx, dmrs)
            snt_root.append(align_root)

        return xml.tostring(snt_root, encoding='utf-8')

    if format == 'text':
        if src and trg or src and align or trg and align:
            raise Exception('Cannot output more than one of source, target, or align at once.')

        if src or trg:
            out_count = len([x for x in [plain, tok, idx, dmrs] if x])
            if out_count != 1:
                raise Exception('Cannout output more than one of plain, tok, idx, or dmrs at once.')

            if src:
                return side_dumps(sentence.source, 'source', format, plain, tok, idx, dmrs)
            else:
                return side_dumps(sentence.target, 'target', format, plain, tok, idx, dmrs)

        elif sid:
            return '{}\t{}'.format(sentence.id, sentence.orig_id)

        elif align:
            raise NotImplementedError('Outputting alignment information to text is not yet implemented.')

    else:
        raise NotImplementedError('Outputting to format %s not implemented.' % format)


def side_dumps(side, side_name, format='xml', plain=True, tok=True, idx=True, dmrs=True):
    if format == 'xml':
        xml_root = xml.Element(side_name)
        xml_root.text = '\n'
        xml_root.tail = '\n'

        if plain and side.plain_text is not None:
            xml_plain = xml.Element('plaintext')
            xml_plain.text = side.plain_text
            xml_plain.tail = '\n'
            xml_root.append(xml_plain)

        if tok and side.tokenized_text is not None:
            xml_tok = xml.Element('tokenized')
            xml_tok.text = ' '.join(side.tokenized_text)
            xml_tok.tail = '\n'
            xml_root.append(xml_tok)

        if idx and side.tokenized_idx is not None:
            xml_idx = xml.Element('idx')
            xml_idx.text = ' '.join([str(x) for x in side.tokenized_idx])
            xml_idx.tail = '\n'
            xml_root.append(xml_idx)

        if dmrs and side.dmrs is not None:
            xml_root.append(side.dmrs)

        return xml_root

    elif format == 'text':

        if plain and side.plain_text is not None:
            return side.plain_text

        if tok and side.tokenized_text is not None:
            return ' '.join(side.tokenized_text)

        if idx and side.tokenized_idx is not None:
            return ' '.join([str(x) for x in side.tokenized_idx])

        if dmrs and side.dmrs is not None:
            return xml.tostring(side.dmrs, encoding='utf-8')

        logging.warn('Output missing.')
        return ''

    else:
        raise NotImplementedError('Outputting to format %s not implemented.' % format)


def align_dumps(align, format='xml', tok=True, idx=True, dmrs=True):
    if format == 'xml':
        xml_root = xml.Element('alignment')
        xml_root.text = '\n'
        xml_root.tail = '\n'

        if tok and align.sttt is not None:
            xml_align = xml.Element('sttt', attrib={'comment': 'string-to-string'})
            xml_align.text = '\n'
            xml_align.tail = '\n'

            xml_align_text = xml.Element('plain')
            xml_align_text.text = '\n'
            xml_align_text.tail = '\n'

            xml_align_index = xml.Element('index')
            xml_align_index.text = '\n'
            xml_align_index.tail = '\n'

            xml_align.append(xml_align_text)
            xml_align.append(xml_align_index)

            for text_align, index_align in align.sttt:
                xml_single_align = xml.Element('align', attrib={'src': text_align[0], 'trg': text_align[1]})
                xml_single_align.tail = '\n'
                xml_align_text.append(xml_single_align)

                xml_single_align = xml.Element('align', attrib={'src': str(index_align[0]), 'trg': str(index_align[1])})
                xml_single_align.tail = '\n'
                xml_align_index.append(xml_single_align)

            xml_root.append(xml_align)

        if tok and align.sgtt is not None:
            xml_align = xml.Element('sgtt', attrib={'comment': 'graph-to-string'})
            xml_align.text = '\n'
            xml_align.tail = '\n'

            xml_align_text = xml.Element('plain')
            xml_align_text.text = '\n'
            xml_align_text.tail = '\n'

            xml_align_index = xml.Element('index')
            xml_align_index.text = '\n'
            xml_align_index.tail = '\n'

            xml_align.append(xml_align_text)
            xml_align.append(xml_align_index)

            for text_align, index_align in align.sgtt:
                xml_single_align = xml.Element('align', attrib={'src': text_align[0], 'trg': ' '.join(text_align[1])})
                xml_single_align.tail = '\n'
                xml_align_text.append(xml_single_align)

                xml_single_align = xml.Element('align', attrib={'src': index_align[0],
                                                                'trg': ' '.join([str(x) for x in index_align[1]])})
                xml_single_align.tail = '\n'
                xml_align_index.append(xml_single_align)

            xml_root.append(xml_align)

        if tok and align.sttg is not None:
            logging.warn('Outputting string-to-graph alignment unsupported.')

        if tok and align.sgtg is not None:
            logging.warn('Outputting graph-to-graph alignment unsupported.')

        return xml_root

    else:
        raise NotImplementedError('Outputting to format %s not implemented.' % format)


def load(fp, format='xml', subset_range=None):
    return loads(fp.read().decode('utf-8'), format, subset_range)


def loads(sent_col_str, format='xml', subset_range=None):
    #if format == 'xml':
    #    sentence_space = '\n\n'
    #else:
    #    sentence_space = '\n'

    sent_strs = ['<sentence' + sent_str for sent_str in sent_col_str.split('<sentence') if sent_str.strip() != '']

    if subset_range is not None:
        sent_strs = sent_strs[subset_range[0] - 1:subset_range[1]]

    sent_col = SentenceCollection()

    for sent_str in sent_strs:
        sent_col.sentences.append(sentence_loads(sent_str, format))

    return sent_col


def sentence_loads(sent_str, format='xml'):
    if format == 'xml':
        try:
            parser = xml.XMLParser(encoding='utf-8')
            sent_xml = xml.fromstring(sent_str.encode('utf8'), parser=parser)
        except xml.ParseError:
            sys.stderr.write(sent_str.encode('utf-8'))
            raise

        source_side = None
        target_side = None
        alignment = None

        for element in sent_xml:

            if element.tag == 'source':
                source_side = side_load(element, format)

            elif element.tag == 'target':
                target_side = side_load(element, format)

            elif element.tag == 'alignment':
                alignment = align_load(element, format)

        return Sentence(sent_xml.attrib.get('id'), source_side, target_side, alignment,
                        orig_id=sent_xml.attrib.get('orig_id'))

    else:
        raise NotImplementedError('Reading format %s not implemented.' % format)


def side_load(side, format='xml'):
    if format == 'xml':

        plain = None
        tok = None
        idx = None
        dmrs = None

        for element in side:

            if element.tag == 'plaintext':
                plain = element.text

            elif element.tag == 'tokenized':
                tok = element.text.split(' ')

            elif element.tag == 'idx':
                idx = [int(x) for x in element.text.split(' ')]

            elif element.tag == 'dmrs':
                dmrs = element if len(element) > 0 else None

        return Side(plain, tok, idx, dmrs)

    else:
        raise NotImplementedError('Reading format %s not implemented.' % format)


def align_load(align, format='xml'):
    if format == 'xml':
        sttt = None
        sgtt = None
        sttg = None
        sgtg = None

        for element in align:

            if element.tag == 'sttt':
                sttt_plain = list()
                sttt_index = list()

                for subelement in element.findall('plain')[0]:
                    if subelement.tag == 'align':
                        sttt_plain.append((subelement.attrib.get('src'), subelement.attrib.get('trg')))

                for subelement in element.findall('index')[0]:
                    if subelement.tag == 'align':
                        sttt_index.append((int(subelement.attrib.get('src')), int(subelement.attrib.get('trg'))))

                sttt = zip(sttt_plain, sttt_index)

            if element.tag == 'sgtt':
                sgtt_plain = list()
                sgtt_index = list()

                for subelement in element.findall('plain')[0]:
                    if subelement.tag == 'align':
                        sgtt_plain.append((subelement.attrib.get('src'), subelement.attrib.get('trg').split(' ')))

                for subelement in element.findall('index')[0]:
                    if subelement.tag == 'align':
                        sgtt_index.append((subelement.attrib.get('src'),
                                           [int(x) for x in subelement.attrib.get('trg').split(' ') if
                                            x.strip() != '']))

                sgtt = zip(sgtt_plain, sgtt_index)

            elif element.tag == 'sttg':
                logging.warn('Reading string-to-graph alignment unsupported.')

            elif element.tag == 'sgtg':
                logging.warn('Reading graph-to-graph alignment unsupported.')

        return Alignment(sttt, sgtt, sttg, sgtg)

    else:
        raise NotImplementedError('Reading format %s not implemented.' % format)


def result_dump(sent_col, fp, format='text', min_id=None, max_id=None, encode=False):
    dump_string = result_dumps(sent_col, format, min_id, max_id)

    if encode:
        fp.write(dump_string.encode('utf-8'))
    else:
        fp.write(dump_string)


def result_dumps(sent_col, format='text', min_id=None, max_id=None):

    if format == 'text':
        pointer = 1
    elif format != 'idx':
        pointer = 0
    else:
        raise NotImplementedError('Outputting results in format %s not supported' % format)

    sent_col = sorted(sent_col, key=lambda x: int(x.id))

    if min_id is not None or max_id is not None:
        sent_str_dict = defaultdict(str)
        for sentence in sent_col:
            sentence_result = ''

            if sentence.result is not None and sentence.result.n_best_list:
                sentence_result = sentence.result.n_best_list[0][pointer]

            sent_str_dict[int(sentence.id)] = sentence_result

        if min_id is None:
            min_id = 0

        if max_id is None:
            max_id = max(sent_str_dict.keys()) + 1

        sent_str_list = [sent_str_dict[sid] for sid in range(min_id, max_id)]

    else:
        sent_str_dict = defaultdict(str)
        for sentence in sent_col:
            if sentence.result is not None and sentence.result.n_best_list:
                sentence_result = sentence.result.n_best_list[0][pointer]
            else:
                sentence_result = ''

            sent_str_dict[int(sentence.id)] = sentence_result
            
        sent_str_list = [sent_str_dict[sid] for sid in sorted(sent_str_dict.keys())]

    return '\n'.join(sent_str_list)
