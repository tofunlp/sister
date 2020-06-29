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
