from unittest import TestCase
from unittest.mock import patch

import numpy as np

from sister.word_embedders import WordEmbedding


class WordEmbeddingCase(TestCase):

    def setUp(self):
        self.words = ['I', 'am', 'a', 'dog', '.']

    def tearDown(self):
        pass

    def test_get_word_vectors_not_implemented(self):
        class Dummy(WordEmbedding):
            def get_word_vector(self, w): ...
        with self.assertRaises(NotImplementedError):
            words = self.words
            Dummy().get_word_vectors(words)

    def test_get_word_vector_not_implemented(self):
        class Dummy(WordEmbedding):
            def get_word_vectors(self, w): ...
        with self.assertRaises(NotImplementedError):
            words = self.words
            Dummy().get_word_vector(words)


class FasttextEmbeddingCase(TestCase):

    def setUp(self):
        embedding_patcher = patch('sister.word_embedders.FasttextEmbedding')
        self.embedding = embedding_patcher.start()(lang='en')
        self.embedding.get_word_vector.return_value = np.random.rand(300)
        self.embedding.get_word_vectors.side_effect = lambda words: np.random.rand(len(words), 300)
        self.embedding_patcher = embedding_patcher

    def tearDown(self):
        self.embedding_patcher.stop()

    def test_get_word_vector(self):
        word = 'test'
        vector = self.embedding.get_word_vector(word)
        self.assertTupleEqual(vector.shape, (300,))
        self.embedding.get_word_vector.assert_called_once_with(word)

    def test_get_word_vectors(self):
        words = ['good', 'test']
        vectors = self.embedding.get_word_vectors(words)
        self.assertTupleEqual(vectors.shape, (len(words), 300))
        self.embedding.get_word_vectors.assert_called_once_with(words)
