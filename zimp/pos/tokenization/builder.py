from enum import Enum

from zimp.pos.tokenization.sentence import SimpleSentenceTokenizer, PunktSentenceTokenizer, EuroParlSentenceTokenizer, \
    CorpusPunktSentenceTokenizer, SpacySentenceTokenizer
from zimp.pos.tokenization.tokenizer import *


class TokenizerStrategy(Enum):
    REGEX = 'REGEX'  # modified sklearn default regex - r"(?u)\b\w\w*\b"
    PYTHON = 'PYTHON'  # use built-in split
    NLTK_BASE = 'NLTK_BASE'
    WORD_PUNKT = 'WORD_PUNKT'  # nltk default regex - r"\w+|[^\w\s]+"
    NLTK_TWEET = 'NLTK_TWEET'  # twitter specifics
    NLTK_NIST = 'NLTK_NIST'  # https://www.nltk.org/api/nltk.tokenize.nist.html
    SPACY = 'SPACY'
    GENSIM = 'GENSIM'


def build_tokenizer(strategy, language):
    if strategy == TokenizerStrategy.REGEX:
        return RegexTokenizer()
    elif strategy == TokenizerStrategy.PYTHON:
        return PythonTokenizer()
    elif strategy == TokenizerStrategy.NLTK_BASE:
        return NltkTokenizer(language)
    elif strategy == TokenizerStrategy.WORD_PUNKT:
        return RegexTokenizer(pattern=r"\w+|[^\w\s]+")
    elif strategy == TokenizerStrategy.NLTK_TWEET:
        return NltkTweetTokenizer()
    elif strategy == TokenizerStrategy.NLTK_NIST:
        return NltkNistTokenizer()
    elif strategy == TokenizerStrategy.SPACY:
        return SpacyTokenizer(language)
    elif strategy == TokenizerStrategy.GENSIM:
        return GensimTokenizer()

    raise AttributeError(f'Strategy {strategy} not yet supported')


class SentenceTokenizerStrategy(Enum):
    SIMPLE = 'SIMPLE'
    PUNKT = 'PUNKT'
    CORPUS_PUNKT = 'CORPUS_PUNKT'
    EUROPARL = 'EUROPARL'
    SPACY = 'SPACY'


def build_sentence_tokenizer(
        strategy: SentenceTokenizerStrategy,
        language: str = None,
        dataset: List[str] = None) -> Tokenizer:
    if strategy == SentenceTokenizerStrategy.SIMPLE:
        return SimpleSentenceTokenizer()
    elif strategy == SentenceTokenizerStrategy.PUNKT:
        return PunktSentenceTokenizer(language)
    elif strategy == SentenceTokenizerStrategy.CORPUS_PUNKT:
        return CorpusPunktSentenceTokenizer(dataset)
    elif strategy == SentenceTokenizerStrategy.EUROPARL:
        return EuroParlSentenceTokenizer(language)
    elif strategy == SentenceTokenizerStrategy.SPACY:
        return SpacySentenceTokenizer(language)
