from unittest import TestCase
from unittest.mock import patch

import numpy as np

from sister.tokenizers import Tokenizer, SimpleTokenizer
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
        embedding_patcher = patch('sister.word_embedders.FasttextEmbedding')
        embedding = embedding_patcher.start()(lang='en')
        embedding.get_word_vector.return_value = np.random.rand(300)
        embedding.get_word_vectors.side_effect = lambda words: np.random.rand(len(words), 300)

        self.sentence_embedding = MeanEmbedding(
            tokenizer=SimpleTokenizer(),
            word_embedder=embedding
        )
        self.addCleanup(embedding_patcher.stop)

    def tearDown(self):
        pass

    def test_embed(self):
        sentence = "I am a dog."
        vector = self.sentence_embedding(sentence)
        self.assertEqual(vector.shape, (300,))
