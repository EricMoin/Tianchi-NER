from collections import Counter
from typing import Generator


class ConllEntity:
    tokens: list[str]
    labels: list[str]

    def __init__(self, tokens: list[str], labels: list[str]):
        self.tokens = tokens
        self.labels = labels


class ConllReader:
    tokens: list[str]
    labels: list[str]

    def __init__(self, conll_file: str):
        self.conll_file = conll_file

    def read(self):
        with open(self.conll_file, 'r', encoding='utf8') as f:
            tokens = []
            labels = []
            for line in f:
                if line == '' or line == '\n':
                    if tokens:
                        yield ConllEntity(tokens, labels)
                        tokens = []
                        labels = []
                else:
                    splits = line.rsplit(' ', 1)
                    tokens.append(splits[0])
                    labels.append(splits[-1].rstrip())
            if tokens:
                yield ConllEntity(tokens, labels)


class MultiConllReader:
    def __init__(self, conll_files: list[str]):
        self.conll_files = conll_files

    def read(self):
        for conll_file in self.conll_files:
            reader = ConllReader(conll_file)
            for entity in reader.read():
                yield entity
