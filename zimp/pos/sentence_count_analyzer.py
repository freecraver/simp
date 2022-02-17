import re
from typing import List

from nltk import sent_tokenize

from zimp.pos.analyzer import SimpleAggregatedAnalyzer
from zimp.pos.tokenization.builder import SentenceTokenizerStrategy, build_sentence_tokenizer


class SentenceCountAnalyzer(SimpleAggregatedAnalyzer):

    def __init__(self,
                 texts: List[str],
                 strategy: SentenceTokenizerStrategy = SentenceTokenizerStrategy.PUNKT,
                 language: str = 'english'):
        """

        :param texts: corpus texts
        :param strategy: strategy to count sentences
        :param language: only used for punkt tokenizer and spacy, for custom models
        """
        super().__init__(texts)
        self.strategy = strategy
        self.language = language
        self.tokenizer = build_sentence_tokenizer(strategy, language, texts)

    def extract_text_metric(self, text: str) -> int:
        return len(self.tokenizer.tokenize_text(text))
