import logging
from abc import ABC, abstractmethod
from typing import List


class BaseSimplifier(ABC):

    def __init__(self):
        if self._can_init_statistics():
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
        if not self._can_init_statistics():
            logging.warning('Statistics are not initialized. Results might be wrong')

        return [self.simplify_text(text) for text in texts]

    def _can_init_statistics(self) -> bool:
        """
        :return: True if simplifier requires no more parameters for calculating statistics, e.g. corpus
        """
        return True

    def load_parameters(self, **kwargs):
        """
        assigns parameters and tries to initialize statistics
        """
        if self._can_init_statistics():
            self._init_statistics()
        else:
            logging.warning('Some parameters are still missing - initialization will not be performed')
