from typing import List


def load_lpc_deu_news_2015_100K_sents() -> List[str]:
    try:
        fp = open("data/lpc/deu_news_2015_100K-sentences.txt", "r")
        X = [s.split("\t")[1].strip() for s in fp.readlines()]
        fp.close()
        return X
    except Exception as err:
        raise Exception(err)
