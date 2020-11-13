# nlptasks
A collection of boilerplate code for different NLP tasks with standardised input/output data types so that it becomes easier to combine NLP tasks with different libraries/models under the hood.

- [Sentence Boundary Disambiguation (SBD)](#sentence-boundary-disambiguation)
- [Word Tokenization](#word-tokenization)
- [Lemmatization](#lemmatization)
- [PoS-Tagging](#pos-tagging)
- [Named Entity Recognition (NER)](#named-entity-recognition)
- [Dependency Relations](#dependency-relations)


## Installation
The `nlptasks` package is available on the [PyPi server](https://pypi.org/project/nlptasks/)

```sh
pip install nlptasks>=0.2.1
```

## Sentence Boundary Disambiguation
**Input:**

- A list of M **documents** as string (data type: `List[str]`)

**Output:**

- A list of K **sentences** as string (data type: `List[str]`)


**Usage:**

```py
from nlptasks.sbd import sbd_factory
docs = [
    "Die Kuh ist bunt. Die Bäuerin mäht die Wiese.", 
    "Ein anderes Dokument: Ganz super! Oder nicht?"]
my_sbd_fn = sbd_factory(name="somajo")
sents = my_sbd_fn(docs)
print(sents)
```

Example output:

```
[
    'Die Kuh ist bunt.', 
    'Die Bäuerin mäht die Wiese.', 
    'Ein anderes Dokument: Ganz super!', 
    'Oder nicht?'
]
```


**Algorithms:**

| Factory `name` | Package | Algorithm | Notes |
|:------:|:-------:|:---------:|:-----:|
| `'spacy-de'` | `de_core_news_lg-2.3.0` | Rule-based tokenization followed by Dependency Parsing for SBD | |
| `'stanza-de'` | `stanza==1.1.*`, `de` | Char-based Bi-LSTM + 1D-CNN Dependency Parser for Tokenization, MWT and SBD | [Qi et. al. (2018)](https://nlp.stanford.edu/pubs/qi2018universal.pdf), [GitHub](https://github.com/stanfordnlp/stanza/tree/master/stanza/models) |
| `'nltk-punkt-de'` | `nltk==3.5`, `german` | Punkt Tokenizer, rule-based | [Kiss and Strunk (2006)](https://www.aclweb.org/anthology/J06-4003.pdf), [Source Code](https://www.nltk.org/_modules/nltk/tokenize/punkt.html) |
| `'somajo-de'` | `SoMaJo==2.1.1`, `de_CMC` | rule-based | [Proisl and Uhrig (2016)](http://aclweb.org/anthology/W16-2607), [GitHub](https://github.com/tsproisl/SoMaJo) |
| `'spacy-rule-de'` | `spacy==2.3.0` | rule-based | [Sentencizer class](https://spacy.io/api/sentencizer) |


Notes:

- Dependency parser based SBDs (e.g. `'spacy'`, `'stanza'`) are more suitable for documents with typos (e.g. `','` instead of `'.'`, `' .'` instead of `'. '`) or missing punctuation.
- Rule-based based SBD algorithms (e.g. `'nltk_punkt'`, `'somajo'`, `'spacy_rule'`) are more suitable for documents that can be assumed error free, i.e. it's very likely that spelling and grammar rules are being followed by the author, e.g. newspaper articles, published books, reviewed articles.


## Word Tokenization
**Input:**

- A list of K **sentences** as string (data type: `List[str]`)

**Output:**

- A list of K **token sequences** (data type: `List[List[str]]`)


**Usage:**

```py
from nlptasks.token import token_factory
sentences = [
    "Die Kuh ist bunt.", 
    "Die Bäuerin mäht die Wiese."]
my_tokenizer_fn = token_factory(name="stanza")
sequences = my_tokenizer_fn(sentences)
print(sequences)
```

Example output

```
[
    ['Die', 'Kuh', 'ist', 'bunt', '.'], 
    ['Die', 'Bäuerin', 'mäht', 'die', 'Wiese', '.']
]
```

**Algorithms:**

| Factory `name` | Package | Algorithm | Notes |
|:------:|:-------:|:---------:|:-----:|
| `'spacy-de'` | `de_core_news_lg-2.3.0` | Rule-based tokenization  | [Docs](https://spacy.io/usage/linguistic-features#tokenization) |
| `'stanza-de'` | `stanza==1.1.*`, `de` | Char-based Bi-LSTM + 1D-CNN Dependency Parser for Tokenization, MWT and SBD | [Qi et. al. (2018)](https://nlp.stanford.edu/pubs/qi2018universal.pdf), [GitHub](https://github.com/stanfordnlp/stanza/tree/master/stanza/models) |



## Lemmatization
**Input:**

- A list of **token sequences** (data type: `List[List[str]]`)

**Outputs:**

- A list of **ID sequences** (data type: `List[List[int]]`)
- Vocabulary with `ID:Lemma` mapping (data type: `List[str]`)


**Usage:**

```py
from nlptasks.lemma import lemma_factory
sequences = [
    ['Die', 'Kuh', 'ist', 'bunt', '.'], 
    ['Die', 'Bäuerin', 'mäht', 'die', 'Wiese', '.']
]
my_lemmatizer_fn = lemma_factory(name="spacy")
idseqs, VOCAB = my_lemmatizer_fn(sequences, n_min_occurence=0)
print(idseqs)
print(VOCAB)
```

Example output

```
[[5, 2, 7, 4, 0], [5, 1, 6, 5, 3, 0]]
['.', 'Bäuerin', 'Kuh', 'Wiese', 'bunt', 'der', 'mähen', 'sein', '[UNK]']
```

**Algorithms:**

| Factory `name` | Package | Algorithm | Notes |
|:------:|:-------:|:---------:|:-----:|
| `'spacy-de'` | `de_core_news_lg-2.3.0` | Rule-based tokenization  | [Docs](https://spacy.io/usage/linguistic-features#tokenization) |
| `'stanza-de'` | `stanza==1.1.*`, `de` | n.a. | [Qi et. al. (2018), Ch. 2.3](https://nlp.stanford.edu/pubs/qi2018universal.pdf), [GitHub](https://github.com/stanfordnlp/stanza/tree/master/stanza/models) |


## PoS-Tagging
**Input:**

- A list of **token sequences** (data type: `List[List[str]]`)

**Outputs:**

- A list of **ID sequences** (data type: `List[List[int]]`)
- Vocabulary with `ID:postag` mapping, i.e. the "tag set" (data type: `List[str]`)


**Usage:**

```py
from nlptasks.pos import pos_factory
sequences = [
    ['Die', 'Kuh', 'ist', 'bunt', '.'], 
    ['Die', 'Bäuerin', 'mäht', 'die', 'Wiese', '.']
]
my_postagger = pos_factory(name="spacy")
idseqs, TAGSET = my_postagger(sequences, maxlen=4)
print(idseqs)
```

Example output

```
[[19, 41, 4, 2], [48, 10, 19, 2]]
```


**Algorithms:**

| Factory `name` | Package | Algorithm | Notes |
|:------:|:-------:|:---------:|:-----:|
| `'spacy-de'` | `de_core_news_lg-2.3.0` | multi-task CNN | [Docs](https://spacy.io/usage/linguistic-features#pos-tagging) |
| `'stanza-de'` | `stanza==1.1.*`, `de` | Bi-LSTM with a) word2vec, b) own embedding layer, c) char-based embedding as input | [Qi et. al. (2018), Ch. 2.2](https://nlp.stanford.edu/pubs/qi2018universal.pdf), [GitHub](https://github.com/stanfordnlp/stanza/tree/master/stanza/models) |
| `'flair-de'` | `flair==0.6.*`, `de-pos-ud-hdt-v0.5.pt` |  | [Docs](https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_2_TAGGING.md#german-models) |



## Named Entity Recognition
The NE-tags without prefix (e.g. `LOC`, `PER`) are mapped with IDs, i.e. `int`.
 
**Input:**

- A list of **token sequences** (data type: `List[List[str]]`)

**Outputs:**

- A list of **ID sequences** (data type: `List[List[int]]`)
- Vocabulary with `ID:nerscheme` mapping (data type: `List[str]`)


**Usage:**

```py
from nlptasks.ner import ner_factory
sequences = [
    ['Die', 'Frau', 'arbeit', 'in', 'der', 'UN', '.'], 
    ['Angela', 'Merkel', 'mäht', 'die', 'Wiese', '.']
]
my_ner = ner_factory(name="spacy")
idseqs, SCHEME = my_ner(sequences)
print(idseqs)
print(SCHEME)
```

Example output

```
[[4, 4, 4, 4, 4, 2, 4], [0, 0, 4, 4, 4, 4]]
['PER', 'LOC', 'ORG', 'MISC', '[UNK]']
```


**Algorithms:**

| Factory `name` | Package | Algorithm | Notes |
|:------:|:-------:|:---------:|:-----:|
| `'flair-multi'` | `flair==0.6.*`, `quadner-large.pt` |  | [Docs](https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_2_TAGGING.md#multilingual-models) |
| `'spacy-de'` | `de_core_news_lg-2.3.0` | multi-task CNN | [Docs](https://spacy.io/usage/linguistic-features#named-entities) |
| `'stanza-de'` | `stanza==1.1.*`, `de` | n.a. | [Docs](https://stanfordnlp.github.io/stanza/available_models.html#available-ner-models), [GitHub](https://github.com/stanfordnlp/stanza/tree/master/stanza/models) |



## NER (Variant 2)
The NER tagger will return NE-tags with IOB-prefix, e.g. `E-LOC`.
Both information are one-hot encoded, i.e. one token (column) can have one or two 1s.

**Input:**

- A list of **token sequences** (data type: `List[List[str]]`)

**Outputs:**

- A list of **index pairs of a logical matrix** (data type: `List[List[Tuple[int, int]]]`)
- A list with with original sequence length
- Numbers of NER-Scheme tags `len(nerscheme)`

**Usage:**

```py
from nlptasks.ner2 import ner2_factory
sequences = [
    ['Die', 'Frau', 'arbeit', 'in', 'der', 'UN', '.'], 
    ['Angela', 'Merkel', 'mäht', 'die', 'Wiese', '.']
]
my_ner = ner2_factory(name="flair-multi")
maskseqs, seqlen, SCHEME = my_ner(sequences)
print(maskseqs)
print(seqlen)
print(SCHEME)
```

Example output

```
[
    [(6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (8, 5), (2, 5), (6, 6)], 
    [(4, 0), (0, 0), (7, 1), (0, 1), (6, 2), (6, 3), (6, 4), (6, 5)]
]
['PER', 'LOC', 'ORG', 'MISC', 'B', 'I', 'O', 'E', 'S']
```

**Algorithms:**

| Factory `name` | Package | Algorithm | Notes |
|:------:|:-------:|:---------:|:-----:|
| `'flair-multi'` | `flair==0.6.*`, `quadner-large.pt` |  | [Docs](https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_2_TAGGING.md#multilingual-models) |


## Dependency Relations
**Input:**

- A list of **token sequences** (data type: `List[List[str]]`)

**Outputs:**

- A list of **index pairs of an adjacency matrix** (data type: `List[List[Tuple[int, int]]]`) for
    - children relations to a token
    - parent relation to a token
- A list with with original sequence length


**Usage:**

```py
from nlptasks.deprel import deprel_factory
sequences = [
    ['Die', 'Kuh', 'ist', 'bunt', '.'], 
    ['Die', 'Bäuerin', 'mäht', 'die', 'Wiese', '.']
]
my_deps = deprel_factory("spacy")
deps_child, deps_parent, seqlens = my_deps(sequences)
print(deps_child)
print(deps_parent)
```

Example output

```
[
    [(0, 1), (1, 2), (3, 2), (4, 2)], 
    [(0, 1), (1, 2), (4, 2), (5, 2), (3, 4)]
]
[
    [(1, 0), (2, 1), (2, 2), (2, 3), (2, 4)], 
    [(1, 0), (2, 1), (2, 2), (4, 3), (2, 4), (2, 5)]
]
```

**Algorithms:**

| Factory `name` | Package | Algorithm | Notes |
|:------:|:-------:|:---------:|:-----:|
| `'spacy-de'` | `de_core_news_lg-2.3.0` |  multi-task CNN | [Docs](https://spacy.io/usage/linguistic-features#dependency-parse) |


# Appendix

### Install a virtual environment

```
python3.6 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-dev.txt --use-feature=2020-resolver
pip install -r requirements.txt --use-feature=2020-resolver
python scripts/nlptasks_downloader.py
bash download_testdata.sh
```

(If your git repo is stored in a folder with whitespaces, then don't use the subfolder `.venv`. Use an absolute path without whitespaces.)

### Python commands

* Jupyter for the examples: `jupyter lab`
* Check syntax: `flake8 --ignore=F401 --exclude=$(grep -v '^#' .gitignore | xargs | sed -e 's/ /,/g')`
* Run Unit Tests: `pytest`
* Upload to PyPi with twine: `python setup.py sdist && twine upload -r pypi dist/*`

### Clean up 

```
find . -type f -name "*.pyc" | xargs rm
find . -type d -name "__pycache__" | xargs rm -r
rm -r .pytest_cache
rm -r .venv
```


### Support
Please [open an issue](https://github.com/ulf1/nlptasks/issues/new) for support.


### Contributing
Please contribute using [Github Flow](https://guides.github.com/introduction/flow/). Create a branch, add commits, and [open a pull request](https://github.com/ulf1/nlptasks/compare/).
