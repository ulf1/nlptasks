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


def test2():  # check pad_idseqs
    sentences = load_lpc_deu_news_2015_100K_sents()
    sentences = sentences[:1000]
    
    seqs_token = token_factory("spacy")(sentences)
    
    seqs_lemma, VOCAB_LEMMA = lemma_factory("spacy")(
        seqs_token, n_min_occurence=20,
        maxlen=32, padding='pre', truncating='pre')
    
    assert len(seqs_token) == len(seqs_lemma)
    assert all([len(seqs) == 32 for seqs in seqs_lemma])
