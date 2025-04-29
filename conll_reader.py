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
