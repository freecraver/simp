import io

import pandas as pd
import requests
from zimp.pos.tokenization.builder import TokenizerStrategy, build_tokenizer
from zimp.readability.metrics import ReadabilityScore
from typing import List

"""
Lively, Bertha A., and Sidney L. Pressey. "A method for measuring the vocabulary burden of textbooks."
Educational administration and supervision 9.7 (1923): 389-398.
"""

cached_ds = {}


class VocabularySizeScore(ReadabilityScore):
    """
    simply calculates number of unique words per text
    """

    def __init__(self,
                 word_tokenizer_strategy: TokenizerStrategy = TokenizerStrategy.NLTK_BASE,
                 lowercase=True,
                 language='english'):
        self.word_tokenizer = build_tokenizer(word_tokenizer_strategy, language)
        self.lowercase = lowercase

    def get_score(self, text: str) -> float:
        tokens = self.word_tokenizer.tokenize_text(text)
        if self.lowercase:
            tokens = [t.lower() for t in tokens]
        return len(set(tokens))


class OutOfVocabularySizeScore(ReadabilityScore):
    """
    calculates number of words not in reference vocabulary, including duplicates
    """

    def __init__(self,
                 word_list=None,
                 word_tokenizer_strategy: TokenizerStrategy = TokenizerStrategy.NLTK_BASE,
                 lowercase=True,
                 language='english'):
        self.word_tokenizer = build_tokenizer(word_tokenizer_strategy, language)
        self.word_list = set(word_list or get_default_word_list(language))
        self.lowercase = lowercase

    def get_score(self, text: str) -> float:
        tokens = self.word_tokenizer.tokenize_text(text)
        if self.lowercase:
            tokens = [t.lower() for t in tokens]
        return len([t for t in tokens if t not in self.word_list])


def get_default_word_list(lang: str) -> List[str]:
    """
    returns top n words for passed language. in memory cache for repeated calls
    """
    suffix = 'en'
    if lang == 'german':
        suffix = 'de'

    cached_texts = cached_ds.get(lang)
    if cached_texts:
        return cached_texts

    ds_url = f'https://raw.githubusercontent.com/freecraver/zimp_resources/main/word_frequencies/top_10000_{suffix}.csv'
    f_stream = requests.get(ds_url).content
    cached_ds[lang] = pd.read_csv(io.StringIO(f_stream.decode('utf-8')), header=None, comment='#').loc[:, 0]
    return cached_ds[lang]