import re
import nltk.tokenize.punkt as punkt
from typing import List

from nltk import sent_tokenize

from zimp.pos.tokenization.sentence_splitter import SentenceSplitter
from zimp.pos.tokenization.tokenizer import Tokenizer, get_spacy_base_model


class SimpleSentenceTokenizer(Tokenizer):

    def __init__(self, pattern=r'[.!?]+'):
        self.pattern = pattern

    def tokenize_text(self, text: str) -> List[str]:
        return list(filter(None, re.split(self.pattern, text)))


class PunktSentenceTokenizer(Tokenizer):

    def __init__(self, language: str = 'english'):
        self.language = language

    def tokenize_text(self, text: str) -> List[str]:
        return sent_tokenize(text, self.language)


class CorpusPunktSentenceTokenizer(Tokenizer):

    def __init__(self, dataset: List[str]):
        self.text = "\n".join(dataset)
        self.tokenizer = punkt.PunktSentenceTokenizer(train_text=self.text)

    def tokenize_text(self, text: str) -> List[str]:
        return self.tokenizer.tokenize(text)


class EuroParlSentenceTokenizer(Tokenizer):

    def __init__(self, language: str = 'english'):
        self.lang_code = self._get_lang_code(language)
        self.splitter = SentenceSplitter(self.lang_code)

    def tokenize_text(self, text: str) -> List[str]:
        return self.splitter.split(text)

    @staticmethod
    def _get_lang_code(language: str):
        if language == 'english':
            return 'en'
        elif language == 'german':
            return 'de'

        raise ValueError(f'EuroParlTokenizer online supports english and german, but received "{language}"')


class SpacySentenceTokenizer(Tokenizer):

    def __init__(self, language: str = 'english'):
        self._base_model = get_spacy_base_model(language)
        self._base_model.add_pipe('sentencizer')

    def tokenize_text(self, text: str) -> List[str]:
        return list(self._base_model(text).sents)
