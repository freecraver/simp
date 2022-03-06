from typing import List

from zimp.pos.countvectorizer_analyzer import CountVectorizerAnalyzer
from zimp.pos.tokenization.builder import TokenizerStrategy, build_tokenizer
from zimp.simplification.simplifier import BaseSimplifier


class SimpleWordSimplifier(BaseSimplifier):

    def __init__(self,
                 lowercase=True,
                 min_length=None,
                 max_length=None,
                 tokenizer_strategy: TokenizerStrategy = TokenizerStrategy.NLTK_BASE,
                 tokenizer_language: str = 'english'):
        self.lowercase = lowercase
        self.min_length = min_length
        self.max_length = max_length
        self.tokenizer = build_tokenizer(tokenizer_strategy, language=tokenizer_language)
        super().__init__()

    def simplify_text(self, text: str) -> str:
        if self.lowercase:
            text = text.lower()

        tokens = []
        for token in self.tokenizer.tokenize_text(text):
            if self.min_length and self.min_length > len(token):
                continue
            if self.max_length and self.max_length < len(token):
                continue
            tokens.append(token)

        return ' '.join(tokens)


class VocabularyWordSimplifier(BaseSimplifier):

    def __init__(self,
                 dataset: List[str] = [],
                 min_term_frequency: float = 1,
                 max_document_frequency: float = 0,
                 lowercase: bool = False,
                 tokenizer_strategy: TokenizerStrategy = TokenizerStrategy.NLTK_BASE,
                 tokenizer_language: str = 'english'):
        """
        :param dataset: corpus used to calculate vocabulary
        :param min_term_frequency: min number of occurrences of word across whole corpus, if a value in the interval
        of ]0,1[ is used, this is considered as a fraction of the total word count, otherwise absolute count
        :param max_document_frequency: maximum number of documents the word is allowed to appear on average
        (multiple occurrences within a single document count too);
        if a value in the interval of ]0,1[ is used, this is considered as a fraction of the total document count
        :param lowercase: convert all texts to lowercase
        :param tokenizer_strategy: strategy used to obtain individual words
        :param tokenizer_language: text-language used at tokenization
        """
        self._dataset = dataset
        self._min_tf = min_term_frequency
        self._max_df = max_document_frequency
        self.min_tf = 1
        self.max_df = None
        self.vocabulary = {}
        self._lowercase = lowercase
        self._tokenizer_language  = tokenizer_language
        self._tokenizer_strategy = tokenizer_strategy
        self._cva = CountVectorizerAnalyzer(
            dataset,
            lowercase=lowercase,
            language=tokenizer_language,
            strategy=tokenizer_strategy)
        super().__init__()

    def _init_statistics(self):
        self.min_tf = 1.0
        df_vocab = self._cva.extract_dataset_metric()
        if self._min_tf < 1:
            self.min_tf = float(self._min_tf * df_vocab.sum())
        elif self._min_tf > 1:
            self.min_tf = self._min_tf

        if not self._max_df:
            self.max_df = None
        elif self._max_df < 1:
            self.max_df = float(self._max_df * len(self._dataset))
        else:
            self.max_df = self._max_df

        df_vocab['count'] = df_vocab['count'].astype(float)
        selection_clause = df_vocab['count'] >= self.min_tf
        if self.max_df:
            selection_clause &= df_vocab['count'] <= self.max_df

        self.vocabulary = set(df_vocab[selection_clause].index)

    def can_init_statistics(self) -> bool:
        return bool(self._dataset)

    def simplify_text(self, text: str) -> str:
        if self._lowercase:
            text = text.lower()

        tokens = self._cva.tokenizer.tokenize_text(text)
        text = ' '.join([token for token in tokens if token in self.vocabulary])

        return text


    def load_parameters(self, dataset: List[str], **kwargs):
        self._dataset = dataset
        self._cva = CountVectorizerAnalyzer(dataset,
                                            lowercase=self._lowercase,
                                            language=self._tokenizer_language,
                                            strategy=self._tokenizer_strategy)
        super().load_parameters(**kwargs)
