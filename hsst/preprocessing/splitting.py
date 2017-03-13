import math
import random

from hsst.utility.representation import SentenceCollection


class BaseSplitter(object):
    def split(self, dataset):
        return dataset


class ProportionSplitter(BaseSplitter):
    def __init__(self, proportions_list, random=False):
        if sum(proportions_list) != 1.0:
            raise Exception('Dataset split proportions must sum to 1.')

        self.proportions_list = proportions_list
        self.random = random

    def split(self, dataset):
        if self.random:
            dataset = list(dataset)
            random.shuffle(dataset)

        sizes = [int(math.ceil(len(dataset) * proportion)) for proportion in self.proportions_list]
        datasets = list()

        dataset_pointer = 0
        for size in sizes[:-1]:
            datasets.append(SentenceCollection(dataset[dataset_pointer:dataset_pointer + size]))
            dataset_pointer += size

        datasets.append(SentenceCollection(dataset[dataset_pointer:]))

        return datasets


class SizeSplitter(BaseSplitter):
    def __init__(self, size_list, random=False):
        self.size_list = size_list
        self.random = random

    def split(self, dataset):
        if sum(self.size_list) != len(dataset):
            raise Exception('Dataset sizes must sum to the total dataset size.')

        if self.random:
            dataset = list(dataset)
            random.shuffle(dataset)

        datasets = list()

        dataset_pointer = 0
        for size in self.size_list[:-1]:
            datasets.append(SentenceCollection(dataset[dataset_pointer:dataset_pointer + size]))
            dataset_pointer += size

        datasets.append(SentenceCollection(dataset[dataset_pointer:]))

        return datasets


def dataset_split(sentence_collection, splitter):
    return splitter.split(sentence_collection)