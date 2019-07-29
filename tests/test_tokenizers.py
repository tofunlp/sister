from unittest import TestCase

from sister import tokenizers


class TokenizerCase(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_tokenize_not_implemented(self):
        class Dummy(tokenizers.Tokenizer):
            pass
        with self.assertRaises(NotImplementedError):
            sentence = 'I am a dog.'
            Dummy().tokenize(sentence)


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


class JapaneseTokenizer(TestCase):

    def setUp(self):
        self.tokenizer = tokenizers.JapaneseTokenizer()

    def tearDown(self):
        pass

    def test_tokenize(self):
        sentence = "私は犬だ。"
        gold = ["私", "は", "犬", "だ", "。"]
        self.assertEqual(
                gold,
                self.tokenizer.tokenize(sentence)
                )
