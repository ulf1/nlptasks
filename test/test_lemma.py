from nlptasks.testdata import load_lpc_deu_news_2015_100K_sents
from nlptasks.token import token_factory
from nlptasks.lemma import lemma_factory
import nlptasks as nt


def test01():
    sentences = load_lpc_deu_news_2015_100K_sents()
    sentences = sentences[:500]
    
    seqs_token = token_factory("spacy")(sentences)
    
    seqs_lemma, VOCAB_LEMMA = lemma_factory("spacy")(
        seqs_token, min_occurrences=20)
    
    assert len(VOCAB_LEMMA) == 61
    assert len(seqs_token) == len(seqs_lemma)


def test02():  # check pad_idseqs
    sentences = load_lpc_deu_news_2015_100K_sents()
    sentences = sentences[:100]
    
    seqs_token = token_factory("spacy")(sentences)
    
    seqs_lemma, VOCAB_LEMMA = lemma_factory("spacy")(
        seqs_token, min_occurrences=20,
        maxlen=32, padding='pre', truncating='pre')
    
    assert len(seqs_token) == len(seqs_lemma)
    assert all([len(seqs) == 32 for seqs in seqs_lemma])

    # lemmatize again with a given VOCAB
    seqs_lemma2, VOCAB_LEMMA2 = lemma_factory("spacy")(
        seqs_token, VOCAB=VOCAB_LEMMA,
        maxlen=32, padding='pre', truncating='pre')

    assert VOCAB_LEMMA2 == VOCAB_LEMMA
    assert seqs_lemma2 == seqs_lemma


def test03():
    sentences = load_lpc_deu_news_2015_100K_sents()
    sentences = sentences[:50]
    
    identifier = "spacy-de"
    model = nt.token.get_model(identifier)
    fn = nt.token.token_factory(identifier)
    seqs_token = fn(sentences, model=model)
    
    identifier = "spacy-de"
    model = nt.lemma.get_model(identifier)
    fn = nt.lemma.lemma_factory(identifier)
    seqs_lemma, VOCAB_LEMMA = fn(seqs_token, min_occurrences=20, model=model)

    assert len(VOCAB_LEMMA) == 6
    assert len(seqs_token) == len(seqs_lemma)


def test11():
    sentences = load_lpc_deu_news_2015_100K_sents()
    sentences = sentences[:500]
    
    seqs_token = token_factory("spacy")(sentences)
    
    seqs_lemma, VOCAB_LEMMA = lemma_factory("stanza-de")(
        seqs_token, min_occurrences=20)
    
    assert len(VOCAB_LEMMA) == 62
    assert len(seqs_token) == len(seqs_lemma)


def test12():  # check pad_idseqs
    sentences = load_lpc_deu_news_2015_100K_sents()
    sentences = sentences[:100]
    
    seqs_token = token_factory("spacy")(sentences)
    
    seqs_lemma, VOCAB_LEMMA = lemma_factory("stanza-de")(
        seqs_token, min_occurrences=20,
        maxlen=32, padding='pre', truncating='pre')
    
    assert len(seqs_token) == len(seqs_lemma)
    assert all([len(seqs) == 32 for seqs in seqs_lemma])

    # lemmatize again with a given VOCAB
    seqs_lemma2, VOCAB_LEMMA2 = lemma_factory("stanza-de")(
        seqs_token, VOCAB=VOCAB_LEMMA,
        maxlen=32, padding='pre', truncating='pre')

    assert VOCAB_LEMMA2 == VOCAB_LEMMA
    assert seqs_lemma2 == seqs_lemma


def test13():
    sentences = load_lpc_deu_news_2015_100K_sents()
    sentences = sentences[:50]
    
    identifier = "stanza-de"
    model = nt.token.get_model(identifier)
    fn = nt.token.token_factory(identifier)
    seqs_token = fn(sentences, model=model)
    
    identifier = "stanza-de"
    model = nt.lemma.get_model(identifier)
    fn = nt.lemma.lemma_factory(identifier)
    seqs_lemma, VOCAB_LEMMA = fn(seqs_token, min_occurrences=20, model=model)

    assert len(VOCAB_LEMMA) == 7
    assert len(seqs_token) == len(seqs_lemma)
