from unittest import TestCase

from sister.word_embedders import WordEmbedding, FasttextEmbedding


class WordEmbeddingCase(TestCase):

    def setUp(self):
        self.words = ['I', 'am', 'a', 'dog', '.']

    def tearDown(self):
        pass

    def test_get_word_vector_not_implemented(self):
        class Dummy(WordEmbedding):
            def get_word_vectors(self, w): ...
        with self.assertRaises(NotImplementedError):
            words = self.words
            Dummy().get_word_vector(words)


class FasttextEmbeddingCase(TestCase):

    def setUp(self):
        # TODO
        # Downloading fasttext and unzipping is too heavy.
        # Needs to be mocked somehow.
        # For now, solved with caching.

        # embedding_patcher = patch('sister.word_embedders.FasttextEmbedding')
        # self.embedding = embedding_patcher.start()(lang='en')
        # self.embedding.get_word_vector.return_value = np.random.rand(300)
        # self.embedding.get_word_vectors.side_effect = lambda words: np.random.rand(len(words), 300)
        # self.embedding_patcher = embedding_patcher
        self.embedding = FasttextEmbedding(lang="en")

    def tearDown(self):
        # self.embedding_patcher.stop()
        pass

    def test_get_word_vector(self):
        word = 'test'
        vector = self.embedding.get_word_vector(word)
        self.assertTupleEqual(vector.shape, (300,))
        # self.embedding.get_word_vector.assert_called_once_with(word)

    def test_get_word_vectors(self):
        words = ['good', 'test']
        vectors = self.embedding.get_word_vectors(words)
        self.assertTupleEqual(vectors.shape, (len(words), 300))
        # self.embedding.get_word_vectors.assert_called_once_with(words)
