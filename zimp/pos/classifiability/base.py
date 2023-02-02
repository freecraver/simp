from abc import ABC, abstractmethod


class DatasetScore(ABC):

    @abstractmethod
    def score(self, X, y):
        """
        scores the difficulty of an NLP task
        :param X: list of texts
        :param y: list of class labels
        :return: a difficulty score, defined by implementing class
        """
        pass