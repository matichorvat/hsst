from hsst.utility.representation import SentenceCollection


class BaseFilter(object):

    def filter(self, sentence):
        return True


class SourceGraphSizeFilter(BaseFilter):

    def __init__(self, max_nodes=20, min_nodes=0):
        self.max = max_nodes if max_nodes > 0 else None
        self.min = min_nodes

    def filter(self, sentence):
        if sentence.source.dmrs is not None:
            count = 0
            for entity in sentence.source.dmrs:
                if entity.tag == 'node':
                    count += 1

            if self.min <= count:
                if self.max is None:
                    return True
                elif count <= self.max:
                    return True

        return False


class SourceTokLength(BaseFilter):

    def __init__(self, max_len=20, min_len=0):
        self.max = max_len
        self.min = min_len

    def filter(self, sentence):
        if sentence.source.tokenized_text is not None:
            return self.min <= len(sentence.source.tokenized_text) <= self.max

        return False


def dataset_filter(sentence_collection, *args):
    filtered_sentence_collection = SentenceCollection()
    for sentence in sentence_collection:
        if all([filterobj.filter(sentence) for filterobj in args]):
            filtered_sentence_collection.add(sentence)

    return filtered_sentence_collection
