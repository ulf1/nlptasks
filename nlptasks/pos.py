from typing import List
import de_core_news_lg as spacy_model
import stanza


def pos_factory(name: str):
    if name == "spacy":
        return pos_spacy_de
    elif name == "stanza":
        return pos_stanza_de
    elif name == "imsnpars_zdl":
        return pos_imsnpars_zdl
    else:
        raise Exception(f"Unknown PoS tagger: '{name}'") 

