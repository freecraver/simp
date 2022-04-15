from abc import ABC, abstractmethod
from typing import List

from zimp.pos.sentence_count_analyzer import SentenceCountAnalyzer
from zimp.pos.wordcount_analyzer import WordCountAnalyzer


class ReadabilityScore(ABC):

    @abstractmethod
    def get_score(self, text: str) -> float:
        pass

    def get_scores(self, texts: List[str]) -> List[float]:
        return [self.get_score(text) for text in texts]


class SentenceLengthScore(ReadabilityScore):

    def __init__(self, word_count_analyzer=None, sentence_count_analyzer=None, language='english'):
        self.wca = word_count_analyzer or WordCountAnalyzer([], language=language)
        self.sca = sentence_count_analyzer or SentenceCountAnalyzer([], language=language)

    def get_score(self, text: str) -> float:
        return self.wca.extract_text_metric(text) / self.sca.extract_text_metric(text)
