from zimp.pos.analyzer import SimpleAggregatedAnalyzer
from zimp.pos.tokenization.builder import build_tokenizer, TokenizerStrategy
from typing import List


class WordCountAnalyzer(SimpleAggregatedAnalyzer):

    def __init__(self,
                 texts: List[str],
                 strategy: TokenizerStrategy = TokenizerStrategy.REGEX,
                 language: str = 'english'):
        """

        :param texts: corpus texts
        :param strategy: strategy to count words
        :param language: only used for nltk-base tokenizer and spacy, for custom rules
        """
        super().__init__(texts)
        self.strategy = strategy
        self.language = language
        self.tokenizer = build_tokenizer(strategy, language)

    def extract_text_metric(self, text: str) -> int:
        return len(self.tokenizer.tokenize_text(text))
