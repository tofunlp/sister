# sister
SISTER (**SI**mple **S**en**T**ence **E**mbedde**R**)


# Installation

```bash
pip install sister
```


# Basic Usage
```python
import sister
sentence_embedding = sister.SentenceEmbedding()

sentence = "I am a dog."
vector = sentence_embedding(sentence)
```


# Supported languages.

- English
- Japanese

In order to support a new language, please implement `Tokenizer` (inheriting `sister.tokenizers.Tokenizer`) and add fastText
pre-trained url to `word_embedders.get_fasttext()` ([List of model urls](https://github.com/facebookresearch/fastText/blob/master/docs/pretrained-vectors.md)).
