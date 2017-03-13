from collections import Counter


class BaseVocab(object):

    def __init__(self):
        self.vocab_freq = Counter()

    def extract(self, dataset):
        for sentence in dataset:
            self.vocab_freq += self.extract_sentence(sentence)

    def extract_sentence(self, sentence):
        pass

    def get_freq(self):
        return self.vocab_freq

    def write_vocab(self, filename):
        with open(filename, 'wb') as fp:
            for item, freq in self.vocab_freq.most_common():
                fp.write('%s\t%d\n' % (item.encode('utf-8'), freq))


class SourceGraphVocab(BaseVocab):

    def __init__(self):
        super(SourceGraphVocab, self).__init__()

    def extract_sentence(self, sentence):
        vocab = Counter()

        if sentence.source.dmrs is None:
            return vocab

        for entity in sentence.source.dmrs:

            if entity.tag == 'node':
                node_label = entity.attrib.get('label')
                if node_label is not None:
                    vocab[node_label] += 1

            elif entity.tag == 'link':
                edge_label = entity.attrib.get('label')
                if edge_label is not None:
                    vocab[edge_label] += 1

        return vocab


class SourceGraphCargVocab(BaseVocab):

    def __init__(self):
        super(SourceGraphCargVocab, self).__init__()

    def extract_sentence(self, sentence):
        vocab = Counter()

        if sentence.source.dmrs is None:
            return vocab

        for entity in sentence.source.dmrs:

            if entity.tag == 'node':
                carg = entity.attrib.get('carg')
                if carg is not None:
                    vocab[carg] += 1

        return vocab


class BaseWMAP(object):

    def __init__(self, existing_wmap_filename=None):
        self.wmap = dict()

        if existing_wmap_filename is not None:
            wmap = self.load_wmap(existing_wmap_filename)
            self.next_id = max(wmap.values()) + 1
        else:
            self.next_id = 0

    def get_wmap(self):
        return self.wmap

    def write_wmap(self, filename):
        inv_wmap = {v: k for k, v in self.wmap.items()}

        with open(filename, 'wb') as fp:
            for word_id, item in sorted(inv_wmap.items()):
                fp.write('%d\t%s\n' % (word_id, item.encode('utf-8')))

    def load_wmap(self, filename):
        wmap = dict()

        with open(filename, 'rb') as fp:
            for line in fp:
                entry = line.split('\t')

                if len(entry) != 2:
                    continue

                wmap[entry[1]] = entry[0]

        return wmap


class SourceGraphWMAP(BaseWMAP):

    def dataset_wmap(self, dataset):
        for sentence in dataset:
            self.sentence_wmap(sentence)

    def sentence_wmap(self, sentence):

        if sentence.source.dmrs is None:
            return

        for entity in sentence.source.dmrs:

            if entity.tag == 'node':
                node_label = entity.attrib.get('label')

                if node_label is not None:
                    if node_label not in self.wmap:
                        self.wmap[node_label] = self.next_id
                        self.next_id += 1

                    entity.attrib['label_idx'] = str(self.wmap[node_label])

            elif entity.tag == 'link':
                edge_label = entity.attrib.get('label')

                if edge_label is not None:
                    if edge_label not in self.wmap:
                        self.wmap[edge_label] = self.next_id
                        self.next_id += 1

                    entity.attrib['label_idx'] = str(self.wmap[edge_label])


def dmrs_vocab(dataset, output_filename):
    vocab_extractor = SourceGraphVocab()
    vocab_extractor.extract(dataset)
    vocab_extractor.write_vocab(output_filename)


def dmrs_idmap(dataset, output_filename, existing_wmap_file=None):
    wmap = SourceGraphWMAP(existing_wmap_file)
    wmap.dataset_wmap(dataset)
    wmap.write_wmap(output_filename)
    return dataset






