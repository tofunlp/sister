<div align="center">
    <br>
    <img src="./docs/img/banner.jpg" width="400"/>
</div>


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
