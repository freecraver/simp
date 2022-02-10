import pandas as pd
import numpy as np

from abc import ABC
from typing import List, Callable
from sklearn.feature_extraction.text import CountVectorizer

from zimp.pos.analyzer import CountAnalyzer
from zimp.pos.tokenization.builder import TokenizerStrategy, build_tokenizer


class BaseCountVectorizerAnalyzer(CountAnalyzer, ABC):
    def __init__(self,
                 texts: List[str],
                 f_tokenize: Callable[[str], List[str]],
                 lowercase: bool):
        vectorizer = CountVectorizer(tokenizer=f_tokenize, lowercase=lowercase)
        super().__init__(texts, vectorizer)

    def extract_dataset_metric(self) -> pd.DataFrame:
        counts = np.asarray(self._get_corpus_counts().sum(axis=0)).flatten()
        return pd.DataFrame(counts, self.count_vectorizer.get_feature_names_out(), columns=['count'])\
            .sort_values(by='count', ascending=False)


class CountVectorizerAnalyzer(BaseCountVectorizerAnalyzer):
    def __init__(self,
                 texts: List[str],
                 strategy: TokenizerStrategy = TokenizerStrategy.REGEX,
                 language: str = 'english',
                 lowercase: bool = True):
        """
       :param texts: corpus texts
       :param strategy: strategy to count words
       :param language: only used for nltk-base tokenizer and spacy, for custom rules
       """
        self.tokenizer = build_tokenizer(strategy, language)
        super().__init__(texts, self.tokenizer.tokenize_text, lowercase)


class SymbolCountVectorizerAnalyzer(BaseCountVectorizerAnalyzer):
    """
    used to obtain counts for unique symbols
    """
    def __init__(self, texts: List[str], remove_whitespace: bool = True, lowercase: bool = False, only_alphabetic: bool = False):
        def f_tokenize(text):
            if remove_whitespace:
                text = ''.join(text.split())  # remove any kind of white space, line breaks, tabs
            chars = list(text)
            if only_alphabetic:
                chars = [c for c in chars if c.isalpha()]
            return chars
        super().__init__(texts, f_tokenize, lowercase)
