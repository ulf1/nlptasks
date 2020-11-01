from nlptasks.sbd import (sbd_spacy_de, sbd_stanza_de)


def test_01():
    target = ["Die Kuh ist bunt.", "Die Bäuerin mäht die Wiese."]
    documents = ["Die Kuh ist bunt. Die Bäuerin mäht die Wiese."]
    sentences = sbd_spacy_de(documents)
    assert sentences == target


def test_11():
    target = ["Die Kuh ist bunt.", "Die Bäuerin mäht die Wiese."]
    documents = ["Die Kuh ist bunt. Die Bäuerin mäht die Wiese."]
    sentences = sbd_stanza_de(documents)
    assert sentences == target
