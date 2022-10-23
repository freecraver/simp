import io
from typing import List

import pandas as pd
import requests

from zimp.readability.lively import OutOfVocabularySizeScore
from zimp.readability.metrics import ReadabilityScore, SentenceLengthScore
from zimp.readability.util import get_freq_score

"""
Dale, Edgar, and Jeanne S. Chall.
"A formula for predicting readability: Instructions."
Educational research bulletin (1948): 37-54.
"""


class DaleChallScore(ReadabilityScore):
    """
    higher score means higher difficulty
    4.9 and below   4-th grade and below
    5.0 to 5.9      5-6th grade
    6.0 to 6.9      7-8th grade
    7.0 to 7.9      9-10th grade
    8.0 to 8.9      11-12th grade
    9.0 to 9.9      13-15th grade (college)
    10.0 and above  16-(college graduate)
    """

    def get_score(self, text: str) -> float:
        d = get_freq_score([text], self.oov.get_filtered_toks) * 100
        s = self.sentence_length_score.get_score(text)
        c = 3.6365

        return d * 0.1579 + s * 0.496 + c

    def __init__(self, sentence_length_score=None, language='english', word_list=None):
        """

        :param sentence_length_score: uses for calculating avg sentence length
        :param language: english|german
        :param word_list: list of known words
        """
        self.sentence_length_score = sentence_length_score or SentenceLengthScore(language=language)
        self.oov = OutOfVocabularySizeScore(language=language, word_list=word_list or get_default_word_list(language))


cached_ds = {}


def get_default_word_list(lang: str) -> List[str]:
    """
    returns top n words for passed language. in memory cache for repeated calls
    """
    suffix = ''
    if lang == 'german':
        suffix = '_de'

    cached_texts = cached_ds.get(lang)
    if cached_texts is not None:
        return cached_texts

    ds_url = f'https://raw.githubusercontent.com/freecraver/zimp_resources/main/dictionaries/dale_chall{suffix}.csv'
    f_stream = requests.get(ds_url).content
    cached_ds[lang] = pd.read_csv(io.StringIO(f_stream.decode('utf-8')), header=None, comment='#').loc[:, 0].tolist()
    return cached_ds[lang]


