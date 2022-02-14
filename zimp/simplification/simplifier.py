from abc import ABC, abstractmethod
from typing import List


class BaseSimplifier(ABC):

    def __init__(self):
        self._init_statistics()

    def _init_statistics(self):
        """
        used to perform initialization steps, like calculating term occurrences
        :return: void
        """
        pass

    @abstractmethod
    def simplify_text(self, text: str) -> str:
        """
        :param text: input string
        :return: simplified text
        """
        pass

    def simplify_dataset(self, texts: List[str]) -> List[str]:
        return [self.simplify_text(text) for text in texts]

    def requires_corpus(self) -> bool:
        """
        :return: True if simplifier requires a dataset before simplification can take place
        """
        return False
