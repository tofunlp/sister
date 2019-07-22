from unittest import TestCase

from sister import Core, SentenceEmbedding


class CoreTextCase(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_main(self):
        name = 'sotaro'
        core = Core()
        self.assertEqual(f'Hello {name}!!!', core.main(name))


class SetenceEmbeddingCase(TestCase):

    def setUp(self):
        self.sentence_embedding = SentenceEmbedding()
        pass

    def tearDown(self):
        pass

    def test_embed(self):
        sentence = "I am a dog."
        vector = self.sentence_embedding.embed(sentence)
        self.assertEqual(vector.shape, (300,))
