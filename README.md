# sister
SISTER (**SI**mple **S**en**T**ence **E**mbedde**R**)


# Installation

```bash
pip install sister
```


# Basic Usage
```python
import sister
sentence_embedding = sister.MeanEmbedding(lang="en")

sentence = "I am a dog."
vector = sentence_embedding(sentence)
```

If you have custom model file by yourself, you can load it too.
(Data Format has to be loadable as [`gensim.models.KeyedVectors`](https://radimrehurek.com/gensim/models/keyedvectors.html) for word2vec model files)

```py
import sister
from sister.word_embedders import Word2VecEmbedding

sentence_embedding = sister.MeanEmbedding(
    lang="ja", Word2VecEmbedding(model_path="/path/to/model")
)

sentence = "I am a dog."
vector = sentence_embedding(sentence)
```


# Supported languages.

- English
- Japanese
- French

In order to support a new language, please implement `Tokenizer` (inheriting `sister.tokenizers.Tokenizer`) and add fastText
pre-trained url to `word_embedders.get_fasttext()` ([List of model urls](https://github.com/facebookresearch/fastText/blob/master/docs/pretrained-vectors.md)).


# Bert models are supported for en, fr, ja (2020-06-29).
Actually Albert for English, CamemBERT for French and BERT for Japanese.  
To use BERT, you need to install sister by `pip install 'sister[bert]'`.

```python
import sister
bert_embedding = sister.BertEmbedding(lang="en")

sentence = "I am a dog."
vector = bert_embedding(sentence)
```

You can also give multiple sentences to it (more efficient).

```python
import sister
bert_embedding = sister.BertEmbedding(lang="en")

sentences = ["I am a dog.", "I want be a cat."]
vectors = bert_embedding(sentences)
```
