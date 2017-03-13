import sys
import xml.etree.ElementTree as xml
from collections import Counter


def split_dmrs_file(content):
    content_split = content.split('<dmrs')
    content_filter = filter(lambda x: x.strip() != '', content_split)
    content_fixed = [('<dmrs' + x).strip() for x in content_filter]
    return content_fixed


def empty(dmrs_xml):
    count = 0
    for child in dmrs_xml:
        return False

    return True


def load_wmap(filename):
    wmap = dict()

    with open(filename, 'rb') as fp:
        for line in fp:
            entry = line.strip().split('\t')

            assert len(entry) == 2

            try:
                wmap[entry[1].decode('utf-8')] = int(entry[0])

            except ValueError:
                pass

    return wmap


def extract_sentence_vcb(xml_sentence, vocab, lc=False):

    for entity in xml_sentence:

        if entity.tag == 'node':
            carg = entity.attrib.get('carg')
            if carg is not None:
                #carg = carg[1:-1]

                if lc:
                    carg = carg.lower()

                if '_' in carg:
                    cargs = [x for x in carg.split('_') if x.strip() != '']
                    for carg in cargs:
                        vocab[carg] += 1
                elif '+' in carg:
                    cargs = [x for x in carg.split('+') if x.strip() != '']
                    for carg in cargs:
                        vocab[carg] += 1

                else:
                    vocab[carg] += 1


def extract_dataset_vcb(filename_in, filename_out, lc=False):
    vocab = Counter()

    with open(filename_in) as fin, open(filename_out, 'wb') as fout:

        dmrs_list = split_dmrs_file(fin.read().decode('utf-8').strip())

        for dmrs in dmrs_list:
            parser = xml.XMLParser(encoding='utf-8')
            dmrs_xml = xml.fromstring(dmrs.encode('utf-8'), parser=parser)
            extract_sentence_vcb(dmrs_xml, vocab, lc=lc)

        for word, count in vocab.most_common():
            fout.write('%s\t%d\n' % (word.encode('utf-8'), count))


def map_sentence(xml_sentence, wmap, lc=False):
    for entity in xml_sentence:

        if entity.tag == 'node':
            carg = entity.attrib.get('carg')

            if carg is not None:
                if lc:
                    carg = carg.lower()

                try:
                    if '_' in carg:
                        cargs = carg.split('_')
                    elif '+' in carg:
                        cargs = carg.split('+')
                    elif ' ' in carg:
                        cargs = carg.split(' ')
                    else:
                        cargs = [carg]

                    carg_idx = ' '.join([str(wmap[x]) for x in cargs if x.strip() != ''])

                    entity.attrib['carg_idx'] = carg_idx

                except ValueError:
                    raise Exception('Missing carg in wmap: %s' % carg)

    return xml_sentence


def map_dataset(filename_in, filename_out, wmap, lc=False):

    wmap = load_wmap(wmap)

    with open(filename_in) as fin, open(filename_out, 'wb') as fout:

        dmrs_list = split_dmrs_file(fin.read().decode('utf-8').strip())

        for dmrs in dmrs_list:
            parser = xml.XMLParser(encoding='utf-8')
            dmrs_xml = xml.fromstring(dmrs.encode('utf-8'), parser=parser)

            if empty(dmrs_xml):
                fout.write('%s\n\n' % dmrs)
            else:
                wdmrs = map_sentence(dmrs_xml, wmap, lc=lc)
                fout.write('%s\n\n' % xml.tostring(wdmrs, encoding='utf-8'))


if __name__ == '__main__':

    filename_in = sys.argv[2]
    filename_out = sys.argv[3]
    lc = True if sys.argv[-1] == 'lc' else False

    if sys.argv[1] == 'extract':
        extract_dataset_vcb(filename_in, filename_out, lc=lc)

    elif sys.argv[1] == 'map':
        wmap = sys.argv[4]
        map_dataset(filename_in, filename_out, wmap, lc=lc)
