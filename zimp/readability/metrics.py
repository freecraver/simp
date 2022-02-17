from abc import ABC
from typing import List

from zimp.pos.tokenization.tokenizer import Tokenizer, NltkTokenizer


class ReadabilityScore(ABC):

    def __init__(self, tokenizer: Tokenizer = NltkTokenizer()):
        self.tokenizer = tokenizer

    def get_score(self, text: str) -> float:
        pass

    def get_scores(self, texts: List[str]) -> List[float]:
        return [self.get_score(text) for text in texts]
