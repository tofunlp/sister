from unittest import TestCase

from sister import MeanEmbedding


class MeanEmbeddingCase(TestCase):

    def setUp(self):
        self.sentence_embedding = MeanEmbedding()
        pass

    def tearDown(self):
        pass

    def test_embed(self):
        sentence = "I am a dog."
        vector = self.sentence_embedding.embed(sentence)
        self.assertEqual(vector.shape, (300,))
