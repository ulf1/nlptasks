from .padding import pad_maskseqs
from typing import List, Tuple
import warnings
import stanza

# UPOS v2, https://universaldependencies.org/u/pos/
UPOS_TAGSET = [
    'ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART',
    'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']

# UD v2 features (162), https://universaldependencies.org/u/feat/index.html
UD2_FEATS = [
    'PronType=Art', 'PronType=Dem', 'PronType=Emp', 'PronType=Exc',
    'PronType=Ind', 'PronType=Int', 'PronType=Neg', 'PronType=Prs',
    'PronType=Rcp', 'PronType=Rel', 'PronType=Tot',
    'NumType=Card', 'NumType=Dist', 'NumType=Frac', 'NumType=Mult',
    'NumType=Ord', 'NumType=Range', 'NumType=Sets',
    'Poss=Yes', 'Reflex=Yes', 'Foreign=Yes', 'Abbr=Yes', 'Typo=Yes',
    'Gender=Com', 'Gender=Fem', 'Gender=Masc', 'Gender=Neut',
    'Animacy=Anim', 'Animacy=Hum', 'Animacy=Inan', 'Animacy=Nhum',
    'NounClass=Bantu1', 'NounClass=Bantu2', 'NounClass=Bantu3',
    'NounClass=Bantu4', 'NounClass=Bantu5', 'NounClass=Bantu6',
    'NounClass=Bantu7', 'NounClass=Bantu8', 'NounClass=Bantu9',
    'NounClass=Bantu10', 'NounClass=Bantu11', 'NounClass=Bantu12',
    'NounClass=Bantu13', 'NounClass=Bantu14', 'NounClass=Bantu15',
    'NounClass=Bantu16', 'NounClass=Bantu17', 'NounClass=Bantu18',
    'NounClass=Bantu19', 'NounClass=Bantu20',
    'Number=Coll', 'Number=Count', 'Number=Dual', 'Number=Grpa',
    'Number=Grpl', 'Number=Inv', 'Number=Pauc', 'Number=Plur', 'Number=Ptan',
    'Number=Sing', 'Number=Tri',
    'Case=Abl', 'Case=Add', 'Case=Ade', 'Case=All', 'Case=Del', 'Case=Ela',
    'Case=Ess', 'Case=Ill', 'Case=Ine', 'Case=Lat', 'Case=Loc', 'Case=Per',
    'Case=Sub', 'Case=Sup', 'Case=Ter', 'Case=Abe', 'Case=Ben', 'Case=Cau',
    'Case=Cmp', 'Case=Cns', 'Case=Com', 'Case=Dat', 'Case=Dis', 'Case=Equ',
    'Case=Gen', 'Case=Ins', 'Case=Par', 'Case=Tem', 'Case=Tra', 'Case=Voc',
    'Case=Abs', 'Case=Acc', 'Case=Erg', 'Case=Nom',
    'Definite=Com', 'Definite=Cons', 'Definite=Def', 'Definite=Ind',
    'Definite=Spec',
    'Degree=Abs', 'Degree=Cmp', 'Degree=Equ', 'Degree=Pos', 'Degree=Sup',
    'VerbForm=Conv', 'VerbForm=Fin', 'VerbForm=Gdv', 'VerbForm=Ger',
    'VerbForm=Inf', 'VerbForm=Part', 'VerbForm=Sup', 'VerbForm=Vnoun',
    'Mood=Adm', 'Mood=Cnd', 'Mood=Des', 'Mood=Imp', 'Mood=Ind', 'Mood=Jus',
    'Mood=Nec', 'Mood=Opt', 'Mood=Pot', 'Mood=Prp', 'Mood=Qot', 'Mood=Sub',
    'Tense=Fut', 'Tense=Imp', 'Tense=Past', 'Tense=Pqp', 'Tense=Pres',
    'Aspect=Hab', 'Aspect=Imp', 'Aspect=Iter', 'Aspect=Perf', 'Aspect=Prog',
    'Aspect=Prosp',
    'Voice=Act', 'Voice=Antip', 'Voice=Bfoc', 'Voice=Cau', 'Voice=Dir',
    'Voice=Inv', 'Voice=Lfoc', 'Voice=Mid', 'Voice=Pass', 'Voice=Rcp',
    'Evident=Fh', 'Evident=Nfh', 'Polarity=Neg', 'Polarity=Pos',
    'Person=0', 'Person=1', 'Person=2', 'Person=3', 'Person=4',
    'Polite=Elev', 'Polite=Form', 'Polite=Humb', 'Polite=Infm',
    'Clusivity=Ex', 'Clusivity=In'
]


def factory(name: str):
    if name == "stanza-de":
        return stanza_de
    else:
        raise Exception(f"Unknown PoS tagger: '{name}'") 


def pos2_factory(name: str):
    warnings.warn(
        "Please call `nlptasks.pos2.factory` instead",
        DeprecationWarning, stacklevel=2)
    return factory(name)


def get_model(name: str):
    """Instantiate the pretrained model outside the SBD function
        so that it only needs to be done once

    Parameters:
    -----------
    name : str
        Identfier of the model

    Example:
    --------
        from nlptasks.ner import pos2
        model = pos2.get_model('stanza-de')
        fn = pos2.factory('stanza-de')
        maskseq, seqlen, SCHEME = fn(docs, model=model)
    """
    if name == "stanza-de":
        return stanza.Pipeline(
            lang='de', processors='tokenize,pos',
            tokenize_pretokenized=True)

    else:
        raise Exception(f"Unknown PoS tagger: '{name}'") 


@pad_maskseqs
def stanza_de(data: List[List[str]], model=None) -> (
        List[List[Tuple[int, int]]], List[int], List[str]):
    """PoS-tagging with stanza for German, returns sparse matrix
        sequences of the UPOS scheme and UD features (UD v2).

    Parameters:
    -----------
    data : List[List[str]]
        List of token sequences

    model (Default: None)
        Preloaded instance of the NLP model. See nlptasks.pos2.get_model

    maxlen : Optional[int] = None
        see @nlptasks.padding.pad_maskseqs

    padding : Optional[str] = 'pre'
        see @nlptasks.padding.pad_maskseqs

    truncating : Optional[str] = 'pre'
        see @nlptasks.padding.pad_maskseqs

    Returns:
    --------
    sequences : List[List[Tuple[int, int]]]
        List of sequences that are sparse mask matrices. The rows indicate
          the scheme.

    seqlens : List[int]
        The original length of each sequence
    
    scheme : List[str]
        The UPOS scheme and UD features scheme combined.
        see nlptasks.pos2.UPOS_TAGSET and nlptasks.pos2.UD2_FEATS
    
    Example:
    --------
        maskseq, seqlen, SCHEME = pos2_stanza_de(tokens)
    """
    # (1) load stanza model
    if not model:
        model = stanza.Pipeline(
            lang='de', processors='tokenize,pos',
            tokenize_pretokenized=True)

    # tag all sequences
    docs = model(data)

    # (2) Define the VOCAB/SCHEME
    SCHEME = UPOS_TAGSET.copy() + UD2_FEATS.copy()

    # (3) Lookup all UPOS and UD feats
    maskseqs = []
    seqlen = []
    for sent in docs.sentences:
        pairs = []
        for colidx, t in enumerate(sent.words):
            # lookup UPOS 
            rowidx = SCHEME.index(t.upos)
            pairs.append((rowidx, colidx))
            # loop over all features
            if t.feats:
                for tag in t.feats.split("|"):
                    rowidx = SCHEME.index(tag)
                    pairs.append((rowidx, colidx))
        maskseqs.append(pairs)
        seqlen.append(len(sent.words))

    # done
    return maskseqs, seqlen, SCHEME
