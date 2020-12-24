from typing import List


def load_lpc_deu_news_2015_100K_sents() -> List[str]:
    """Load LPC sentences corpus, German news, year 2015, 100k examples

    Sources:
    --------
    - https://wortschatz.uni-leipzig.de/en/download/german
    """
    try:
        fp = open("data/lpc/deu_news_2015_100K-sentences.txt", "r")
        X = [s.split("\t")[1].strip() for s in fp.readlines()]
        fp.close()
        return X
    except Exception as err:
        raise Exception(err)
