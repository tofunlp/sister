from typing import List
import numpy as np

from sister.tokenizers import Tokenizer, SimpleTokenizer, JapaneseTokenizer
from sister.word_embedders import WordEmbedding, FasttextEmbedding


class SentenceEmbedding(object):

    def __init__(
            self,
            tokenizer: Tokenizer,
            word_embedder: WordEmbedding) -> None:
        self.tokenizer = tokenizer
        self.word_embedder = word_embedder

    def embed(self, sentence: str) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, sentence: str) -> np.ndarray:
        raise NotImplementedError


class MeanEmbedding(SentenceEmbedding):

    def __init__(
            self,
            lang: str = 'en',
            tokenizer: Tokenizer = None,
            word_embedder: WordEmbedding = None) -> None:
        tokenizer = tokenizer or {"en": SimpleTokenizer(),
                                  "fr": SimpleTokenizer(),
                                  "ja": JapaneseTokenizer()}[lang]
        word_embedder = word_embedder or FasttextEmbedding(lang)
        super().__init__(tokenizer, word_embedder)

    def embed(self, sentence: str) -> np.ndarray:
        tokens = self.tokenizer.tokenize(sentence)
        vectors = self.word_embedder.get_word_vectors(tokens)
        return np.mean(vectors, axis=0)

    def __call__(self, sentence: str) -> np.ndarray:
        return self.embed(sentence)


class BertEmbedding:

    def __init__(
            self,
            lang: str = 'en',
            ):
        try:
            from transformers import BertJapaneseTokenizer, AlbertTokenizer, CamembertTokenizer
            from transformers import AlbertModel, CamembertModel, BertModel
        except ImportError:
            msg = "importing bert dep failed."
            msg += "\n try to install sister by `pip install sister[bert]`."
            raise ImportError(msg)

        if lang == "en":
            tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")
            model = AlbertModel.from_pretrained("albert-base-v2")
        elif lang == "fr":
            tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
            model = CamembertModel.from_pretrained("camembert-base")
        elif lang == "ja":
            tokenizer = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
            model = BertModel.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")

        self.tokenizer = tokenizer
        self.model = model

    def embed(self, sentences: List[str]):
        try:
            import torch
        except ImportError:
            msg = "importing bert dep failed."
            msg += "\n try to install sister by `pip install sister[bert]`."
            raise ImportError(msg)

        tokens = self.tokenizer.batch_encode_plus(sentences, pad_to_max_length=True, add_special_tokens=True)
        input_ids = torch.tensor(tokens["input_ids"])[:, :512]
        vector = self.model(input_ids)[0][:, 0, :].detach().numpy()
        return vector

    def __call__(self, sentences: List[str]):

        if isinstance(sentences, str):
            sentences = [sentences]
        hasone = len(sentences) == 1

        vecs = self.embed(sentences)
        if hasone:
            vecs = vecs.reshape(-1)

        return vecs
