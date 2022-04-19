from collections import defaultdict

from zimp.pos.countvectorizer_analyzer import CountVectorizerAnalyzer
from typing import List, DefaultDict


def build_word_vocab(
        texts: List[str] = None,
        max_n=None,
        count_vectorizer_analyzer: CountVectorizerAnalyzer = None) -> List[str]:
    count_vectorizer_analyzer = count_vectorizer_analyzer or CountVectorizerAnalyzer(texts)
    top_words = count_vectorizer_analyzer.extract_dataset_metric().index
    if max_n is None:
        return top_words.to_list()
    return top_words[:max_n].to_list()


def build_word_support(
        texts: List[str] = None,
        max_n=None,
        count_vectorizer_analyzer: CountVectorizerAnalyzer = None) -> DefaultDict[str, int]:
    top_words = build_word_vocab(texts, max_n, count_vectorizer_analyzer)
    return word_list_to_support(top_words)


def word_list_to_support(words: List[str] = None) -> DefaultDict[str, int]:
    """

    :param words: words sorted by descending frequency
    :return: dictionary mapping each word to its support index (credit value), the higher the value the more frequent
             the word, oov words are assigned the value 0
    """
    support_dict = defaultdict(lambda: 0)
    for idx, word in enumerate(words[::-1]):
        support_dict[word] = idx+1
    return support_dict

