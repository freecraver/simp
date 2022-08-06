import io

import pandas as pd
import requests

from zimp.pos.countvectorizer_analyzer import CountVectorizerAnalyzer
from zimp.pos.tokenization.builder import TokenizerStrategy, build_tokenizer
from zimp.readability.metrics import ReadabilityScore
from typing import List

from zimp.readability.vocab import build_word_support, word_list_to_support

"""
Lively, Bertha A., and Sidney L. Pressey. "A method for measuring the vocabulary burden of textbooks."
Educational administration and supervision 9.7 (1923): 389-398.
Bailin, A., & Grafstein, A. (2016). Readability: Text and context. Springer.
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
        s, _ = self.get_filtered_toks(text)
        return len(s)

    def get_filtered_toks(self, text: str):
        tokens = self.word_tokenizer.tokenize_text(text)
        if self.lowercase:
            tokens = [t.lower() for t in tokens]
        return [t for t in tokens if t not in self.word_list], tokens


class LivelyScore(ReadabilityScore):
    """
    calculates the average credit value (support index) of words in a text for a reference corpus
    the higher the score the more easy a text is to read
    out-of-vocabulary words are counted twice to have higher impact
    """

    def __init__(self,
                 word_list=None,
                 reference_texts=None,
                 max_n=None,
                 count_vectorizer_analyzer=None,
                 language='english',
                 lowercase=True
                 ):
        self.lowercase = lowercase
        self.count_vectorizer_analyzer = count_vectorizer_analyzer or CountVectorizerAnalyzer(
            reference_texts, lowercase=lowercase)

        # build lookup dict for words
        if reference_texts is not None:
            self.word_support_dict = build_word_support(reference_texts, max_n, count_vectorizer_analyzer)
        elif word_list:
            self.word_support_dict = word_list_to_support(word_list)
        else:
            word_list = get_default_word_list(language)
            self.word_support_dict = word_list_to_support(word_list)

    def get_score(self, text: str) -> float:
        tokens = self.count_vectorizer_analyzer.tokenizer.tokenize_text(text)
        if self.lowercase:
            tokens = [t.lower() for t in tokens]

        # adapted incremental averaging
        support_avg = 0.0
        cnt = 0.0
        for token in tokens:
            token_support = self.word_support_dict[token]
            cnt += 1
            support_avg = support_avg + (token_support-support_avg)/cnt
            if token_support == 0:
                # update twice -> compact formulation non-trivial, feel free to adapt :)
                cnt += 1
                support_avg = support_avg + (token_support - support_avg) / cnt

        return support_avg


def get_default_word_list(lang: str) -> List[str]:
    """
    returns top n words for passed language. in memory cache for repeated calls
    """
    suffix = 'en'
    if lang == 'german':
        suffix = 'de'

    cached_texts = cached_ds.get(lang)
    if cached_texts is not None:
        return cached_texts

    ds_url = f'https://raw.githubusercontent.com/freecraver/zimp_resources/main/word_frequencies/top_10000_{suffix}.csv'
    f_stream = requests.get(ds_url).content
    cached_ds[lang] = pd.read_csv(io.StringIO(f_stream.decode('utf-8')), header=None, comment='#').loc[:, 0]
    return cached_ds[lang]