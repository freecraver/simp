import pandas as pd
import numpy as np

from abc import ABC, abstractmethod
from sklearn.feature_extraction.text import CountVectorizer
from typing import List


class Analyzer(ABC):
    """
    used to extract text metrics from a dataset
    """

    def __init__(self, texts: List[str]):
        self.texts = texts

    @abstractmethod
    def extract_dataset_metric(self) -> pd.DataFrame:
        pass


class SimpleAggregatedAnalyzer(Analyzer, ABC):
    """
    extract scalar metric value from each text and counts those scalars across the dataset
    """

    def extract_dataset_metric(self) -> pd.DataFrame:
        index, counts = np.unique([self.extract_text_metric(text) for text in self.texts], return_counts=True)
        return pd.DataFrame(index=index, data=counts, columns=['count']).sort_values(by='count', ascending=False)

    @abstractmethod
    def extract_text_metric(self, text: str) -> int:
        pass


class CountAnalyzer(Analyzer, ABC):
    """
    base for analyzers which operate on word counts
    """

    def __init__(self, texts: List[str], count_vectorizer: CountVectorizer = None):
        super(CountAnalyzer, self).__init__(texts)
        self.count_vectorizer = count_vectorizer