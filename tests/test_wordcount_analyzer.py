import unittest

from zimp.pos.wordcount_analyzer import WordCountAnalyzer
from zimp.pos.tokenization.builder import TokenizerStrategy

texts = [
    'How many words are in this sentence?',
    'How do you handle compound-words in the U.S.?',
    'Ehm,do you handle missing spaces?No?',
    '@username213: Twitter  vocabulary of @otheruser12 miGhta bedifferent,hu? xD <3',
    "Why don't you handle clitics with apostrophes differently?"
]


class WordCountAnalyzerTest(unittest.TestCase):

    def test_regex_word_count(self):
        df_res = WordCountAnalyzer(texts, strategy=TokenizerStrategy.REGEX).extract_dataset_metric()
        self.assertEqual({7: 2, 9: 1, 10: 2}, df_res.to_dict()['count'])

    def test_nltk_regex_word_count(self):
        df_res = WordCountAnalyzer(texts, strategy=TokenizerStrategy.WORD_PUNKT).extract_dataset_metric()
        self.assertEqual({8: 1, 10: 1, 11: 1, 13: 1, 16: 1}, df_res.to_dict()['count'])

    def test_split_word_count(self):
        df_res = WordCountAnalyzer(texts, strategy=TokenizerStrategy.PYTHON).extract_dataset_metric()
        self.assertEqual({5: 1, 7: 1, 8: 2, 9: 1}, df_res.to_dict()['count'])

    def test_nltk_base_word_count_en(self):
        df_res = WordCountAnalyzer(texts, strategy=TokenizerStrategy.NLTK_BASE, language='english')\
            .extract_dataset_metric()
        self.assertEqual({8: 1, 9: 1, 10: 2, 16: 1}, df_res.to_dict()['count'])

    def test_nltk_base_word_count_de(self):
        # 'U.S.?' is treated differently with german nltk tokenize
        df_res = WordCountAnalyzer(texts, strategy=TokenizerStrategy.NLTK_BASE, language='german')\
            .extract_dataset_metric()
        self.assertEqual({8: 1, 10: 3, 16: 1}, df_res.to_dict()['count'])

    def test_twitter_word_count(self):
        df_res = WordCountAnalyzer(texts, strategy=TokenizerStrategy.NLTK_TWEET).extract_dataset_metric()
        self.assertEqual({8: 1, 9: 1, 10: 1, 12: 1, 13: 1}, df_res.to_dict()['count'])

    def test_nist_word_count(self):
        df_res = WordCountAnalyzer(texts, strategy=TokenizerStrategy.NLTK_NIST).extract_dataset_metric()
        self.assertEqual({8: 1, 9: 1, 10: 1, 12: 1, 16: 1}, df_res.to_dict()['count'])

    def test_spacy_word_count_en(self):
        df_res = WordCountAnalyzer(texts, strategy=TokenizerStrategy.SPACY, language='english')\
            .extract_dataset_metric()
        self.assertEqual({8: 2, 10: 1, 11: 1, 14: 1}, df_res.to_dict()['count'])

    def test_spacy_word_count_de(self):
        df_res = WordCountAnalyzer(texts, strategy=TokenizerStrategy.SPACY, language='german')\
            .extract_dataset_metric()
        self.assertEqual({8: 1, 9: 2, 10: 1, 14: 1}, df_res.to_dict()['count'])

    def test_gensim_word_count(self):
        df_res = WordCountAnalyzer(texts, strategy=TokenizerStrategy.GENSIM).extract_dataset_metric()
        self.assertEqual({7: 2, 9: 2, 10: 1}, df_res.to_dict()['count'])