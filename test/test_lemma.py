from nlptasks.testdata import load_lpc_deu_news_2015_100K_sents
from nlptasks.token import token_factory
from nlptasks.lemma import lemma_factory


def test01():
    sentences = load_lpc_deu_news_2015_100K_sents()
    sentences = sentences[:5000]
    
    seqs_token = token_factory("spacy")(sentences)
    
    seqs_lemma, VOCAB_LEMMA = lemma_factory("spacy")(
        seqs_token, n_min_occurence=20)
    
    assert len(VOCAB_LEMMA) == 422  # 6284
    assert len(seqs_token) == len(seqs_lemma)


def test02():  # check pad_idseqs
    sentences = load_lpc_deu_news_2015_100K_sents()
    sentences = sentences[:1000]
    
    seqs_token = token_factory("spacy")(sentences)
    
    seqs_lemma, VOCAB_LEMMA = lemma_factory("spacy")(
        seqs_token, n_min_occurence=20,
        maxlen=32, padding='pre', truncating='pre')
    
    assert len(seqs_token) == len(seqs_lemma)
    assert all([len(seqs) == 32 for seqs in seqs_lemma])



def test11():
    sentences = load_lpc_deu_news_2015_100K_sents()
    sentences = sentences[:5000]
    
    seqs_token = token_factory("spacy")(sentences)
    
    seqs_lemma, VOCAB_LEMMA = lemma_factory("stanza-de")(
        seqs_token, n_min_occurence=20)
    
    assert len(VOCAB_LEMMA) == 422
    assert len(seqs_token) == len(seqs_lemma)


def test12():  # check pad_idseqs
    sentences = load_lpc_deu_news_2015_100K_sents()
    sentences = sentences[:1000]
    
    seqs_token = token_factory("spacy")(sentences)
    
    seqs_lemma, VOCAB_LEMMA = lemma_factory("stanza-de")(
        seqs_token, n_min_occurence=20,
        maxlen=32, padding='pre', truncating='pre')
    
    assert len(seqs_token) == len(seqs_lemma)
    assert all([len(seqs) == 32 for seqs in seqs_lemma])
