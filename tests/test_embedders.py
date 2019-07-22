from unittest import TestCase

from sister.word_embedders import FasttextEmbedding


class FasttextEmbeddingCase(TestCase):

    def setUp(self):
        self.embedding = FasttextEmbedding(lang="en")

    def tearDown(self):
        pass

    def test_get_word_vector(self):
        word = 'test'
        vector = self.embedding.get_word_vector(word)
        self.assertEqual(vector.shape, (300,))

    def test_get_word_vectors(self):
        words = ['good', 'test']
        vectors = self.embedding.get_word_vectors(words)
        self.assertEqual(vectors.shape, (len(words), 300))
