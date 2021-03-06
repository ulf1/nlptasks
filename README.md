[![PyPI version](https://badge.fury.io/py/nlptasks.svg)](https://badge.fury.io/py/nlptasks)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4284804.svg)](https://doi.org/10.5281/zenodo.4284804)

# nlptasks
A collection of boilerplate code for various NLP tasks with standardized input and output data types to make it easier to combine NLP tasks with different libraries and models. The main focus lies on the **German** language, multi-lingual models that covers German and its dialects. Only rule-based algorithms or pre-trained models that do not require training are used.

Please consult the [appendix](#appendix) for [installation instructions](#installation).


# Usage
In the following 

**Table of Contents**

- [Sentence Boundary Disambiguation (SBD)](#sentence-boundary-disambiguation)
- [Word Tokenization](#word-tokenization)
- [Lemmatization](#lemmatization)
- Part-of-Speech (PoS) Tagging
    - [PoS tags to ID sequences](#pos-tagging---id-sequences)
    - [PoS tags and Morphological Features to mask sequences](#pos-tagging---mask-sequences)
- Named Entity Recognition (NER)
    - [NER-tags to ID sequences](#named-entity-recognition---id-sequences)
    - [NER-tags and Chunks to mask sequences](#named-entity-recognition---mask-sequences)
- Dependency Relations
    - [Parent Node ID and relation type](#dependency-relations---parents)
    - [Children Nodes of a token](#dependency-relations---children)
    - [Trees as mask indices](#dependency-relations---trees)
- [Meta Information](#meta-information)


## Sentence Boundary Disambiguation
SBD is about splitting a text into sentences (Synonyms: sentence splitting, sentence segmentation, or sentence boundary detection).

**Input:**

- A list of M **documents** as string (data type: `List[str]`)

**Output:**

- A list of K **sentences** as string (data type: `List[str]`)


**Usage:**

```py
import nlptasks as nt
import nlptasks.sbd
docs = [
    "Die Kuh ist bunt. Die Bäuerin mäht die Wiese.", 
    "Ein anderes Dokument: Ganz super! Oder nicht?"]
myfn = nt.sbd.factory(name="somajo")
sents = myfn(docs)
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
| `'spacy-de'` | `de_core_news_lg-2.3.0` | Rule-based tokenization followed by Dependency Parsing for SBD | [10.5281/zenodo.1212303](https://doi.org/10.5281/zenodo.1212303) |
| `'stanza-de'` | `stanza==1.1.*`, `de` | Char-based Bi-LSTM + 1D-CNN Dependency Parser for Tokenization, MWT and SBD | [Qi et. al. (2018)](https://nlp.stanford.edu/pubs/qi2018universal.pdf), [GitHub](https://github.com/stanfordnlp/stanza/tree/master/stanza/models), [10.18653/v1/2020.acl-demos.14](http://dx.doi.org/10.18653/v1/2020.acl-demos.14) |
| `'nltk-punkt-de'` | `nltk==3.5`, `german` | Punkt Tokenizer, rule-based | [Kiss and Strunk (2006)](https://www.aclweb.org/anthology/J06-4003.pdf), [Source Code](https://www.nltk.org/_modules/nltk/tokenize/punkt.html), [10.1162/coli.2006.32.4.485](http://dx.doi.org/10.1162/coli.2006.32.4.485) |
| `'somajo-de'` | `SoMaJo==2.1.1`, `de_CMC` | rule-based | [Proisl and Uhrig (2016)](http://aclweb.org/anthology/W16-2607), [GitHub](https://github.com/tsproisl/SoMaJo), [10.18653/v1/W16-2607](http://dx.doi.org/10.18653/v1/W16-2607) |
| `'spacy-rule-de'` | `spacy==2.3.0` | rule-based | [Sentencizer class](https://spacy.io/api/sentencizer), [10.5281/zenodo.1212303](https://doi.org/10.5281/zenodo.1212303) |


**Notes:**

- Dependency parser based SBDs (e.g. `'spacy'`, `'stanza'`) are more suitable for documents with typos (e.g. `','` instead of `'.'`, `' .'` instead of `'. '`) or missing punctuation.
- Rule-based based SBD algorithms (e.g. `'nltk_punkt'`, `'somajo'`, `'spacy_rule'`) are more suitable for documents that can be assumed error free, i.e. it's very likely that spelling and grammar rules are being followed by the author, e.g. newspaper articles, published books, reviewed articles.


## Word Tokenization
A word consists of one or multiples characters, and has a meaning. Depending on the language, the rules for setting the **word boundaries** have phonetic, orthographic, morphological, syntactic, semantic or other reasons. 
In alphabetic languages like German, the space is usually used to mark a word boundary. That's why *whitespace tokenization* (e.g. `mystring.split(" ")`) is widespread quick hack among coders.

**Input:**

- A list of K **sentences** as string (data type: `List[str]`)

**Output:**

- A list of K **token sequences** (data type: `List[List[str]]`)


**Usage:**

```py
import nlptasks as nt
import nlptasks.token
sentences = [
    "Die Kuh ist bunt.", 
    "Die Bäuerin mäht die Wiese."]
myfn = nt.token.factory(name="stanza-de")
sequences = myfn(sentences)
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
| `'spacy-de'` | `de_core_news_lg-2.3.0` | Rule-based tokenization  | [Docs](https://spacy.io/usage/linguistic-features#tokenization), [10.5281/zenodo.1212303](https://doi.org/10.5281/zenodo.1212303) |
| `'stanza-de'` | `stanza==1.1.*`, `de` | Char-based Bi-LSTM + 1D-CNN Dependency Parser for Tokenization, MWT and SBD | [Qi et. al. (2018)](https://nlp.stanford.edu/pubs/qi2018universal.pdf), [GitHub](https://github.com/stanfordnlp/stanza/tree/master/stanza/models), [10.18653/v1/2020.acl-demos.14](http://dx.doi.org/10.18653/v1/2020.acl-demos.14) |



## Lemmatization
A lemma is the basic form or canonical form of a **set of words** (e.g. a lemma and its flexions). In dictionaries, lemmata are used as **headwords**, and thus, referred to as citation form or dictionary form.

**Input:**

- A list of **token sequences** (data type: `List[List[str]]`)

**Outputs:**

- A list of **ID sequences** (data type: `List[List[int]]`)
- Vocabulary with `ID:Lemma` mapping (data type: `List[str]`)


**Usage:**

```py
import nlptasks as nt
import nlptasks.lemma
sequences = [
    ['Die', 'Kuh', 'ist', 'bunt', '.'], 
    ['Die', 'Bäuerin', 'mäht', 'die', 'Wiese', '.']
]
myfn = nt.lemma.factory(name="spacy")
idseqs, VOCAB = myfn(sequences, min_occurrences=0)
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
| `'spacy-de'` | `de_core_news_lg-2.3.0` | Rule-based tokenization  | [Docs](https://spacy.io/usage/linguistic-features#tokenization), [10.5281/zenodo.1212303](https://doi.org/10.5281/zenodo.1212303) |
| `'stanza-de'` | `stanza==1.1.*`, `de` | n.a. | [Qi et. al. (2018), Ch. 2.3](https://nlp.stanford.edu/pubs/qi2018universal.pdf), [GitHub](https://github.com/stanfordnlp/stanza/tree/master/stanza/models), [10.18653/v1/2020.acl-demos.14](http://dx.doi.org/10.18653/v1/2020.acl-demos.14) |


## PoS-Tagging - ID Sequences
With PoS tagging (Part-of-Speech) every word in a sentence is assigned a grammatical attribute.
The list of grammatical attributes is called **tagset**.

The routines `nlptasks.pos` will run a PoS tagger on each word token of a sequences, and return ID sequences, whereas the IDs map to the PoS tagset.

**Input:**

- A list of **token sequences** (data type: `List[List[str]]`)

**Outputs:**

- A list of **ID sequences** (data type: `List[List[int]]`)
- Vocabulary with `ID:postag` mapping, i.e. the "tag set" (data type: `List[str]`)


**Usage:**

```py
import nlptasks as nt
import nlptasks.pos
sequences = [
    ['Die', 'Kuh', 'ist', 'bunt', '.'], 
    ['Die', 'Bäuerin', 'mäht', 'die', 'Wiese', '.']
]
myfn = nt.pos.factory(name="spacy")
idseqs, TAGSET = myfn(sequences, maxlen=4)
print(idseqs)
```

Example output

```
[[19, 41, 4, 2], [48, 10, 19, 2]]
```


**Algorithms:**

| Factory `name` | Package | Algorithm | Notes |
|:------:|:-------:|:---------:|:-----:|
| `'spacy-de'` | `de_core_news_lg-2.3.0` | multi-task CNN | [Docs](https://spacy.io/usage/linguistic-features#pos-tagging), [10.5281/zenodo.1212303](https://doi.org/10.5281/zenodo.1212303) |
| `'stanza-de'` | `stanza==1.1.*`, `de` | Bi-LSTM with a) word2vec, b) own embedding layer, c) char-based embedding as input | [Qi et. al. (2018), Ch. 2.2](https://nlp.stanford.edu/pubs/qi2018universal.pdf), [GitHub](https://github.com/stanfordnlp/stanza/tree/master/stanza/models), [10.18653/v1/2020.acl-demos.14](http://dx.doi.org/10.18653/v1/2020.acl-demos.14) |
| `'flair-de'` | `flair==0.6.*`, `de-pos-ud-hdt-v0.5.pt` |  | [Docs](https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_2_TAGGING.md#german-models) |
| `'someweta-de'` | `1.7.1` | Perceptron |  [Proisl (2018)](http://www.lrec-conf.org/proceedings/lrec2018/pdf/49.pdf), [Docs](https://github.com/tsproisl/SoMeWeTa#usage) |
| `'someweta-web-de'` | `1.7.1` | Perceptron |  [Proisl (2018)](http://www.lrec-conf.org/proceedings/lrec2018/pdf/49.pdf), [Docs](https://github.com/tsproisl/SoMeWeTa#usage) |



## PoS-Tagging - Mask Sequences
The PoS tagger returns UPOS and UD feats (v2) for a token, e.g. `"DET"` and `"Case=Gen|Definite=Def|Gender=Neut|Number=Sing|PronType=Art"`. All information are boolean encoded, i.e. one token (column) can have one or more 1s.

**Input:**

- A list of **token sequences** (data type: `List[List[str]]`)

**Outputs:**

- A list of **index pairs of a logical matrix** (data type: `List[List[Tuple[int, int]]]`)
- A list with with original sequence length
- Combined UPOS and UD feats Scheme

**Usage:**

```py
import nlptasks as nt
import nlptasks.pos2
sequences = [
    ['Die', 'Frau', 'arbeit', 'in', 'der', 'UN', '.'], 
    ['Angela', 'Merkel', 'mäht', 'die', 'Wiese', '.']
]
myfn = nt.pos2.factory(name="stanza-de")
maskseqs, seqlen, SCHEME = myfn(sequences)
print(maskseqs)
print(seqlen)
print(SCHEME)
```

Example output

```
[
    [
        (5, 0), (112, 0), (115, 0), (41, 0), (77, 0), (17, 0), (7, 1),
        ...
        (11, 5), (100, 5), (41, 5), (77, 5), (12, 6)
    ],
    [
        (11, 0), (112, 0), (41, 0), (77, 0), (11, 1), (112, 1), (41, 1), 
        ... 
        (17, 3), (7, 4), (110, 4), (41, 4), (77, 4), (12, 5)]
    ]
[7, 6]
['ADJ', 'ADP', ... 'VERB', 'X', 'PronType=Art', ..., 'Clusivity=In']
```

**Algorithms:**

| Factory `name` | Package | Algorithm | Notes |
|:------:|:-------:|:---------:|:-----:|
| `'stanza-de'` | `stanza==1.1.*`, `de` | Bi-LSTM with a) word2vec, b) own embedding layer, c) char-based embedding as input | [Qi et. al. (2018), Ch. 2.2](https://nlp.stanford.edu/pubs/qi2018universal.pdf), [GitHub](https://github.com/stanfordnlp/stanza/tree/master/stanza/models), [10.18653/v1/2020.acl-demos.14](http://dx.doi.org/10.18653/v1/2020.acl-demos.14) |



## Named Entity Recognition - ID Sequences
The NE-tags without prefix (e.g. `LOC`, `PER`) are mapped with IDs, i.e. `int`.
 
**Input:**

- A list of **token sequences** (data type: `List[List[str]]`)

**Outputs:**

- A list of **ID sequences** (data type: `List[List[int]]`)
- Vocabulary with `ID:nerscheme` mapping (data type: `List[str]`)


**Usage:**

```py
import nlptasks as nt
import nlptasks.ner
sequences = [
    ['Die', 'Frau', 'arbeit', 'in', 'der', 'UN', '.'], 
    ['Angela', 'Merkel', 'mäht', 'die', 'Wiese', '.']
]
myfn = nt.ner.factory(name="spacy-de")
idseqs, SCHEME = myfn(sequences)
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
| `'spacy-de'` | `de_core_news_lg-2.3.0` | multi-task CNN | [Docs](https://spacy.io/usage/linguistic-features#named-entities), [10.5281/zenodo.1212303](https://doi.org/10.5281/zenodo.1212303) |
| `'stanza-de'` | `stanza==1.1.*`, `de` | n.a. | [Docs](https://stanfordnlp.github.io/stanza/available_models.html#available-ner-models), [GitHub](https://github.com/stanfordnlp/stanza/tree/master/stanza/models), [10.18653/v1/2020.acl-demos.14](http://dx.doi.org/10.18653/v1/2020.acl-demos.14) |



## Named Entity Recognition - Mask Sequences
The NER tagger will return NE-tags with IOB-prefix, e.g. `E-LOC`.
Both information are boolean encoded, i.e. one token (column) can have one or two 1s.

**Input:**

- A list of **token sequences** (data type: `List[List[str]]`)

**Outputs:**

- A list of **index pairs of a logical matrix** (data type: `List[List[Tuple[int, int]]]`)
- A list with with original sequence length
- NER-Scheme tags

**Usage:**

```py
import nlptasks as nt
import nlptasks.ner2
sequences = [
    ['Die', 'Frau', 'arbeit', 'in', 'der', 'UN', '.'], 
    ['Angela', 'Merkel', 'mäht', 'die', 'Wiese', '.']
]
myfn = nt.ner2.factory(name="flair-multi")
maskseqs, seqlen, SCHEME = myfn(sequences)
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


## Dependency Relations - Parents
In CoNLL-U, spacy, stanza, etc. the `head` attribute refers to the parent node of a token, 
i.e. it's an adjacency list.

**Input:**

- A list of **token sequences** (data type: `List[List[str]]`)

**Outputs:**

- A list of **index pairs of an adjacency matrix** (data type: `List[List[Tuple[int, int]]]`) for parent relation to a token
- A list with with original sequence length


**Usage:**

```py
import nlptasks as nt
import nlptasks.dephead
sequences = [
    ['Die', 'Kuh', 'ist', 'bunt', '.'], 
    ['Die', 'Bäuerin', 'mäht', 'die', 'Wiese', '.']
]
myfn = nt.dephead.factory("stanza-de")
maskseqs, seqlens = myfn(
    sequences, maxlen=4, padding='pre', truncating='pre')
print(maskseqs)
print(seqlens)
```

Example output

```
[
    [
        (45, 0), (46, 1), (46, 2), (46, 3), (46, 4),
        (19, 0), (36, 1), (43, 2), (27, 3), (32, 4)
    ],
    [
        (45, 0), (46, 1), (46, 2), (48, 3), (46, 4), (46, 5),
        (19, 0), (36, 1), (43, 2), (19, 3), (21, 4), (32, 5)
    ]
]
[5, 6]
```

**Algorithms:**

| Factory `name` | Package | Algorithm | Notes |
|:------:|:-------:|:---------:|:-----:|
| `'spacy-de'` | `de_core_news_lg-2.3.0` |  multi-task CNN | [Docs](https://spacy.io/usage/linguistic-features#dependency-parse), [10.5281/zenodo.1212303](https://doi.org/10.5281/zenodo.1212303) |
| `'stanza-de'` | `stanza==1.1.*`, `de` | n.a. | [Docs](https://stanfordnlp.github.io/stanza/available_models.html#available-ner-models), [GitHub](https://github.com/stanfordnlp/stanza/tree/master/stanza/models), [10.18653/v1/2020.acl-demos.14](http://dx.doi.org/10.18653/v1/2020.acl-demos.14) |



## Dependency Relations - Children 

**Input** 

- A list of **token sequences** (data type: `List[List[str]]`)

**Outputs** 

- A list of **index pairs of an adjacency matrix** (data type: `List[List[Tuple[int, int]]]`) for children relations to a token.
- A list with with original sequence length

**Usage:**

```py
import nlptasks as nt
import nlptasks.depchild
sequences = [
    ['Die', 'Kuh', 'ist', 'bunt', '.'], 
    ['Die', 'Bäuerin', 'mäht', 'die', 'Wiese', '.']
]
myfn = nt.depchild.factory("spacy-de")
maskseqs, seqlens = myfn(
    sequences, maxlen=4, padding='pre', truncating='pre')
print(maskseqs)
print(seqlens)
```

Example output

```
[
    [(0, 1), (1, 2), (3, 2), (4, 2)], 
    [(0, 1), (1, 2), (4, 2), (5, 2), (3, 4)]
]
[5, 6]
```

**Algorithms:**

| Factory `name` | Package | Algorithm | Notes |
|:------:|:-------:|:---------:|:-----:|
| `'spacy-de'` | `de_core_news_lg-2.3.0` |  multi-task CNN | [Docs](https://spacy.io/usage/linguistic-features#dependency-parse), [10.5281/zenodo.1212303](https://doi.org/10.5281/zenodo.1212303) |


## Dependency Relations - Trees
Most **syntax dependency parsers** return tree representations, usually adjacency list based tree. The `nlptasks.deptree` submodule uses the `treesimi` package (DOI: [10.5281/zenodo.4321304](http://doi.org/10.5281/zenodo.4321304)) to extract subtrees and other tree patterns, which are indexed for a mask vector.


**Input:**

- A list of **token sequences** (data type: `List[List[str]]`)

**Outputs:**

- For each sentences, a list of mask indices (data type: `List[List[int]]`)
- Vocabulary with `ID:Hash` mapping (data type: `List[str]`)


**Usage:**

```py
import nlptasks as nt
import nlptasks.deptree
sequences = [
    ['Die', 'Kuh', 'ist', 'bunt', '.'], 
    ['Die', 'Bäuerin', 'mäht', 'die', 'Wiese', ',', 'aber', 'mit', 'Extra', '.']
]
myfn = nt.deptree.factory("stanza-de")
indices, VOCAB = myfn(sequences)
print(indices)

masked, _ = myfn(sequences, VOCAB=VOCAB, return_mask=True)
print(masked)
```

**Example output**

Each index represents a specific tree or subtree pattern that you can find inside a dependency tree.

```
[
    [0, 1, 2, 3, 4], 
    [5, 1, 2, 6, 7, 4, 8, 9]
]

[
    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 1, 1, 1, 1, 1]
]
```

**Algorithms:**

| Factory `name` | Package | Algorithm | Notes |
|:------:|:-------:|:---------:|:-----:|
| `'spacy-de'` | `de_core_news_lg-2.3.0` |  multi-task CNN | [Docs](https://spacy.io/usage/linguistic-features#dependency-parse), [10.5281/zenodo.1212303](https://doi.org/10.5281/zenodo.1212303) |
| `'stanza-de'` | `stanza==1.1.*`, `de` | n.a. | [Docs](https://stanfordnlp.github.io/stanza/available_models.html#available-ner-models), [GitHub](https://github.com/stanfordnlp/stanza/tree/master/stanza/models), [10.18653/v1/2020.acl-demos.14](http://dx.doi.org/10.18653/v1/2020.acl-demos.14) |


## Meta Information
In order to retrieve the model's meta information, call

```py
import nlptasks as nt
import nlptasks.meta
modelmeta = nt.meta.get(name='nltk-punkt-de', module='sbd')
print(modelmeta)
```

The meta information could be stored next the annotated text data for various database management purposes (e.g. reproducibility, detect changed results due to version changes, compliance with license conditions, etc.)

```sh
{
    'pypi': {
        'name': 'nltk', 'version': '3.5', 'licence': 'Apache-2',
        'isbn': '9780596516499'}, 
    'model': {
        'name': 'punkt', 'file': 'nltk_data/tokenizers/punkt/PY3/german.pickle',
        'modified': '2013-08-23T04:10:02', 'licence': 'Apache-2',
        'doi': '10.1162/coli.2006.32.4.485'}
}
```


# Appendix

## Installation
The `nlptasks` package is available on the [PyPi server](https://pypi.org/project/nlptasks/)

```sh
pip install nlptasks>=0.3.0
```

### Install a virtual environment

```
python3.6 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-dev.txt
pip install -r requirements.txt
python scripts/nlptasks_downloader.py
bash download_testdata.sh
```

(If your git repo is stored in a folder with whitespaces, then don't use the subfolder `.venv`. Use an absolute path without whitespaces.)

### Python commands

* Jupyter for the examples: `jupyter lab`
* Check syntax: `flake8 --ignore=F401 --exclude=.venv,data,.github,.pytest_cache,__pycache__` or `flake8 --ignore=F401 --exclude=$(grep -v '^#' .gitignore | xargs | sed -e 's/ /,/g')`
* Run Unit Tests: `pytest` or `py.test --profile`
* Upload to PyPi with twine: `python setup.py sdist && twine upload -r pypi dist/*`

Some unit test are excluded from pytest due to troubles (e.g. memory, timeout) to run these with CI tools (e.g. Github Actions). You can these manually as follows

```sh
# run all files "test/test_*.py"
py.test --profile
# excluded unit tests
py.test test/excluded_pos.py --profile
```

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
