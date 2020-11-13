import flair
from typing import Union, List


# flair Sentence hack
class FlairSentence(flair.data.Sentence):
    def __init__(
        self,
        text: Union[str, List[str]] = None,
        use_tokenizer: Union[bool, flair.data.Tokenizer] = True,
        language_code: str = None,
        start_position: int = None
    ):
        """
        Same as flair.data.Sentence except
        :param text: original string (sentence), or a list of string tokens (words)
        """
        super(flair.data.Sentence, self).__init__()

        self.tokens: List[Token] = []

        self._embeddings: Dict = {}

        self.language_code: str = language_code

        self.start_pos = start_position
        self.end_pos = (
            start_position + len(text) if start_position is not None else None
        )

        if isinstance(use_tokenizer, flair.data.Tokenizer):
            tokenizer = use_tokenizer
        elif hasattr(use_tokenizer, "__call__"):
            from flair.tokenization import TokenizerWrapper
            tokenizer = TokenizerWrapper(use_tokenizer)
        elif type(use_tokenizer) == bool:
            from flair.tokenization import SegtokTokenizer, SpaceTokenizer
            tokenizer = SegtokTokenizer() if use_tokenizer else SpaceTokenizer()
        else:
            raise AssertionError("Unexpected type of parameter 'use_tokenizer'. " +
                                 "Parameter should be bool, Callable[[str], List[Token]] (deprecated), Tokenizer")

        # if text is passed, instantiate sentence with tokens (words)
        if text is not None:
            if isinstance(text, (list, tuple)):
                [self.add_token(self._restore_windows_1252_characters(token))
                 for token in text]
            else:
                text = self._restore_windows_1252_characters(text)
                [self.add_token(token) for token in tokenizer.tokenize(text)]

        # log a warning if the dataset is empty
        if text == "":
            log.warning(
                "Warning: An empty Sentence was created! Are there empty strings in your dataset?"
            )

        self.tokenized = None
