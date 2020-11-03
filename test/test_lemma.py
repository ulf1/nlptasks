from nlptasks.testdata import load_lpc_deu_news_2015_100K_sents
from nlptasks.token import token_factory
from nlptasks.lemma import lemma_factory


def test1():
    sentences = load_lpc_deu_news_2015_100K_sents()
    
    seqs_token = token_factory("spacy")(sentences)
    
    seqs_lemma, VOCAB_LEMMA = lemma_factory("spacy")(
        seqs_token, n_min_occurence=20)
    
    assert len(VOCAB_LEMMA) == 6284
    assert len(seqs_token) == len(seqs_lemma)

