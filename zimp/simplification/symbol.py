import re

from zimp.pos.countvectorizer_analyzer import SymbolCountVectorizerAnalyzer
from zimp.simplification.simplifier import BaseSimplifier
from typing import List

alpha_regex = re.compile('[^a-zA-Z ]')


class SimpleCharacterSimplifier(BaseSimplifier):

    def __init__(self, lowercase=True, only_alphabetic=False, remove_duplicate_whitespace=True):
        self.lowercase = lowercase
        self.only_alphabetic = only_alphabetic
        self.remove_duplicate_whitespace = remove_duplicate_whitespace
        super().__init__()

    def simplify_text(self, text: str) -> str:
        if self.lowercase:
            text = text.lower()
        if self.only_alphabetic:
            text = alpha_regex.sub(' ', text)
        if self.remove_duplicate_whitespace:
            text = ' '.join(text.split())

        return text


class VocabularyCharacterSimplifier(BaseSimplifier):

    def __init__(self,
                 dataset: List[str] = [],
                 min_term_frequency: float = 1,
                 min_document_frequency: float = 0,
                 remove_duplicate_whitespace=True,
                 lowercase: bool = False):
        """
        :param dataset: corpus used to calculate vocabulary
        :param min_term_frequency: min number of occurrences of symbol across whole corpus, if a value in the interval
        of ]0,1[ is used, this is considered as a fraction of the total character count, otherwise absolute count
        :param min_document_frequency: only considered if min_term_frequency=1; minimum number of documents the
        character must appear on average (multiple occurrences within a single document count too);
        if a value in the interval of ]0,1[ is used, this is considered as a fraction of the total document count
        :param lowercase: convert all texts to lowercase
        """
        self._dataset = dataset
        self._min_tf = min_term_frequency
        self._min_df = min_document_frequency
        self.min_tf = 1
        self.vocabulary = {}
        self._lowercase = lowercase
        self._remove_duplicate_whitespace = remove_duplicate_whitespace
        self._scva = SymbolCountVectorizerAnalyzer(dataset, remove_whitespace=False, lowercase=lowercase)
        super().__init__()

    def _init_statistics(self):
        self.min_tf = 1
        df_vocab = self._scva.extract_dataset_metric()
        if self._min_tf < 1:
            self.min_tf = int(self._min_tf * df_vocab.sum())
        elif self._min_tf > 1:
            self.min_tf = self._min_tf
        elif self._min_df < 1:
            self.min_tf = int(self._min_df * len(self._dataset))
        else:
            self.min_tf = self._min_df

        self.vocabulary = set(df_vocab[df_vocab['count'] >= self.min_tf].index)

    def can_init_statistics(self) -> bool:
        return bool(self._dataset)

    def simplify_text(self, text: str) -> str:
        if self._lowercase:
            text = text.lower()

        substitute_char = ' ' if self._remove_duplicate_whitespace else ''
        text = ''.join([(c if c in self.vocabulary else substitute_char) for c in text])

        if self._remove_duplicate_whitespace:
            text = ' '.join(text.split())

        return text

    def load_parameters(self, dataset: List[str], **kwargs):
        self._dataset = dataset
        self._scva = SymbolCountVectorizerAnalyzer(dataset, remove_whitespace=False, lowercase=self._lowercase)
        super().load_parameters(**kwargs)


