# nlptasks
A collection of boilerplate code for different NLP tasks with standardised input/output data types so that it becomes easier to combine NLP tasks with different libraries/models under the hood.


## NLP Tasks
- [Sentence Boundary Disambiguation (SBD)](#sentence-boundary-disambiguation)


### Sentence Boundary Disambiguation
Input:

- Data type: `List[str]`
- A list of M **documents** as string

Output:

- Data type: `List[str]`
- A list of K **sentences** as string


Algorithms:

| Factory `name` | Package | Algorithm | Notes |
|:------:|:-------:|:---------:|:-----:|
| `'spacy'` | `de_core_news_lg-2.3.0` | Rule-based tokenization followed by Dependency Parsing for SBD | |
| `'stanza'` | `stanza==1.1.*`, `de` | Char-based Bi-LSTM + 1D-CNN Dependency Parser for Tokenization, MWT and SBD | [Qi et. al. (2018)](https://nlp.stanford.edu/pubs/qi2018universal.pdf), [GitHub](https://github.com/stanfordnlp/stanza/tree/master/stanza/models) |
| `'nltk_punkt'` | `nltk==3.5`, `german` | Punkt Tokenizer, rule-based | [Kiss and Strunk (2006)](https://www.aclweb.org/anthology/J06-4003.pdf), [Source Code](https://www.nltk.org/_modules/nltk/tokenize/punkt.html) |
| `'somajo'` | `SoMaJo==2.1.1`, `de_CMC` | rule-based | [Proisl and Uhrig (2016)](http://aclweb.org/anthology/W16-2607), [GitHub](https://github.com/tsproisl/SoMaJo) |
| `'spacy_rule'` | `spacy==2.3.0` | rule-based | [Sentencizer class](https://spacy.io/api/sentencizer) |


## Appendix

### Install a virtual environment

```
python3.8 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-dev.txt
pip install -r requirements.txt
bash download.sh
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
