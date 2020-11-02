from typing import List
import de_core_news_lg as spacy_model
import stanza


def ner_factory(name: str):
    if name == "spacy":
        return ner_spacy_de
    elif name == "stanza":
        return ner_stanza_de
    elif name == "flair":
        return ner_flair_de
    else:
        raise Exception(f"Unknown dependency parser: '{name}'") 

