from typing import List
from pathlib import Path

from fasttext import load_model
import numpy as np

from sister.download import cached_download, cached_unzip


def get_fasttext(lang: str = "en"):
    # Download.
    urls = {
            "en": "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.simple.zip"
            }
    path = cached_download(urls[lang])
    path = Path(path)
    dirpath = path.parent / 'fasttext'
    cached_unzip(path, dirpath)

    print("Loading model...")
    model = load_model(str(dirpath / 'wiki.simple.bin'))
    return model


class WordEmbedding(object):

    def get_word_vector(self, word: str) -> np.ndarray:
        raise NotImplementedError

    def get_word_vectors(self, words: List[str]) -> np.ndarray:
        raise NotImplementedError


class FasttextEmbedding(WordEmbedding):

    def __init__(self, lang: str = "en") -> None:
        model = get_fasttext(lang)
        self.model = model

    def get_word_vector(self, word: str) -> np.ndarray:
        return self.model.get_word_vector(word)

    def get_word_vectors(self, words: List[str]) -> np.ndarray:
        vectors = []
        for word in words:
            vectors.append(self.get_word_vector(word))
        return np.array(vectors)
