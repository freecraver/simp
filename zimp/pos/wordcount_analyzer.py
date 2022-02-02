from enum import Enum

from zimp.pos.analyzer import SimpleAggregatedAnalyzer
from zimp.pos.tokenization.tokenizer import RegexTokenizer, PythonTokenizer, NltkTokenizer, NltkTweetTokenizer, \
    NltkNistTokenizer, SpacyTokenizer, TextBlobTokenizer, GensimTokenizer
from typing import List


class WordCountStrategy(Enum):
    REGEX = 'REGEX'  # modified sklearn default regex - r"(?u)\b\w\w*\b"
    PYTHON = 'PYTHON'  # use built-in split
    NLTK_BASE = 'NLTK_BASE'
    WORD_PUNKT = 'WORD_PUNKT'  # nltk default regex - r"\w+|[^\w\s]+"
    NLTK_TWEET = 'NLTK_TWEET'  # twitter specifics
    NLTK_NIST = 'NLTK_NIST'  # https://www.nltk.org/api/nltk.tokenize.nist.html
    SPACY = 'SPACY'
    TEXTBLOB = 'TEXTBLOB'
    GENSIM = 'GENSIM'


class WordCountAnalyzer(SimpleAggregatedAnalyzer):

    def __init__(self,
                 texts: List[str],
                 strategy: WordCountStrategy = WordCountStrategy.REGEX,
                 language: str = 'english'):
        """

        :param texts: corpus texts
        :param strategy: strategy to count words
        :param language: only used for nltk-based tokenizers, for customer rules
        """
        super().__init__(texts)
        self.strategy = strategy
        self.language = language
        self.tokenizer = self.build_tokenizer()

    def build_tokenizer(self):
        if self.strategy == WordCountStrategy.REGEX:
            return RegexTokenizer()
        elif self.strategy == WordCountStrategy.PYTHON:
            return PythonTokenizer()
        elif self.strategy == WordCountStrategy.NLTK_BASE:
            return NltkTokenizer(self.language)
        elif self.strategy == WordCountStrategy.WORD_PUNKT:
            return RegexTokenizer(pattern=r"\w+|[^\w\s]+")
        elif self.strategy == WordCountStrategy.NLTK_TWEET:
            return NltkTweetTokenizer()
        elif self.strategy == WordCountStrategy.NLTK_NIST:
            return NltkNistTokenizer()
        elif self.strategy == WordCountStrategy.SPACY:
            return SpacyTokenizer()
        elif self.strategy == WordCountStrategy.TEXTBLOB:
            return TextBlobTokenizer()
        elif self.strategy == WordCountStrategy.GENSIM:
            return GensimTokenizer()

        raise AttributeError(f'Strategy {self.strategy} not yet supported')

    def extract_text_metric(self, text: str) -> int:
        return len(self.tokenizer.tokenize_text(text))
