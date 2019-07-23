from unittest import TestCase

from sister.tokenizers import Tokenizer
from sister.word_embedders import WordEmbedding
from sister import MeanEmbedding
from sister.core import SentenceEmbedding


class SentenceEmbeddingCase(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_embed_not_implemented(self):
        class Dummy(SentenceEmbedding):
            def __call__(self, s): ...
        sentence = 'I am a dog.'
        with self.assertRaises(NotImplementedError):
            Dummy(Tokenizer(), WordEmbedding()).embed(sentence)

    def test_call_not_implemented(self):
        class Dummy(SentenceEmbedding):
            def embed(self, s): ...
        sentence = 'I am a dog.'
        with self.assertRaises(NotImplementedError):
            Dummy(Tokenizer(), WordEmbedding()).__call__(sentence)


class MeanEmbeddingCase(TestCase):

    def setUp(self):
        self.sentence_embedding = MeanEmbedding()

    def tearDown(self):
        pass

    def test_embed(self):
        sentence = "I am a dog."
        vector = self.sentence_embedding(sentence)
        self.assertEqual(vector.shape, (300,))
