from nlptasks.testdata import load_lpc_deu_news_2015_100K_sents


def test1():
    sentences = load_lpc_deu_news_2015_100K_sents()
    target = 99270
    assert len(sentences) == target
