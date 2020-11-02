from typing import List
import de_core_news_lg as spacy_model
import stanza


def lemma_factory(name: str):
    if name == "spacy":
        return lemma_spacy_de
    elif name == "stanza":
        return lemma_stanza_de
    else:
        raise Exception(f"Unknown lemmatizer: '{name}'") 

