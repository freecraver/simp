import pandas as pd
import numpy as np

from typing import List
from sklearn.feature_extraction.text import CountVectorizer

from zimp.pos.analyzer import CountAnalyzer
from zimp.pos.tokenization.builder import TokenizerStrategy, build_tokenizer


class CountVectorizerAnalyzer(CountAnalyzer):
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
        vectorizer = CountVectorizer(tokenizer=self.tokenizer.tokenize_text, lowercase=lowercase)
        super().__init__(texts, vectorizer)

    def extract_dataset_metric(self) -> pd.DataFrame:
        counts = np.asarray(self._get_corpus_counts().sum(axis=0)).flatten()
        return pd.DataFrame(counts, self.count_vectorizer.get_feature_names(), columns=['count'])\
            .sort_values(by='count', ascending=False)
