from unittest import TestCase

from sister import tokenizers


class SimpleTokenizerCase(TestCase):

    def setUp(self):
        self.tokenizer = tokenizers.SimpleTokenizer()

    def tearDown(self):
        pass

    def test_tokenize(self):
        sentence = 'I would like to visit Japan.'
        gold = [
                'I',
                'would',
                'like',
                'to',
                'visit',
                'Japan',
                '.'
                ]
        tokens = self.tokenizer.tokenize(sentence)
        self.assertEqual(gold, tokens)
