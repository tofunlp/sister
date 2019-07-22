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
