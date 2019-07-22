from typing import List


class Tokenizer(object):

    def tokenize(self, sentence: str) -> List[str]:
        raise NotImplementedError


class SimpleTokenizer(Tokenizer):

    def __init__(self):
        self.replace_tokens = [
                ('.', ' .'),
                ('?', ' ?'),
                ('!', ' !'),
                ('(', ' ( '),
                (')', ' ) '),
                ]

    def tokenize(self, sentence: str) -> List[str]:
        for replace_token in self.replace_tokens:
            sentence = sentence.replace(*replace_token)
        return sentence.split()
