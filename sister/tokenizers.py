from typing import List
from janome.tokenizer import Tokenizer as JanomeTokenizer


class Tokenizer(object):

    def tokenize(self, sentence: str) -> List[str]:
        raise NotImplementedError


class SimpleTokenizer(Tokenizer):

    def __init__(self):
        self.replace_tokens = str.maketrans({
            '.': ' .',
            '?': ' ?',
            '!': ' !',
            '(': ' ( ',
            ')': ' ) ',
        })

    def tokenize(self, sentence: str) -> List[str]:
        return sentence.translate(self.replace_tokens).split()


class JapaneseTokenizer(Tokenizer):

    def __init__(self):
        self.model = JanomeTokenizer()

    def tokenize(self, sentence: str) -> List[str]:
        return self.model.tokenize(sentence, wakati=True)
