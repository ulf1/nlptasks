from typing import List
import de_core_news_lg as spacy_model
import stanza


def deprel_factory(name: str):
    if name == "spacy":
        return deprel_spacy_de
    elif name == "stanza":
        return deprel_stanza_de
    elif name == "imsnpars_zdl":
        return deprel_imsnpars_zdl
    else:
        raise Exception(f"Unknown dependency parser: '{name}'") 

