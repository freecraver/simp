import unittest

from zimp.pos.wordcount_analyzer import WordCountAnalyzer, WordCountStrategy

texts = [
    'How many words are in this sentence?',
    'How do you handle compound-words in the U.S.?',
    'Ehm,do you handle missing spaces?No?',
    '@username213: Twitter  vocabulary of @otheruser12 miGhta bedifferent,hu? xD <3'
]


class WordCountAnalyzerTest(unittest.TestCase):

    def test_regex_word_count(self):
        df_res = WordCountAnalyzer(texts, strategy=WordCountStrategy.REGEX).extract_dataset_metric()
        self.assertEqual({7: 2, 10: 2}, df_res.to_dict()['count'])

    def test_nltk_regex_word_count(self):
        df_res = WordCountAnalyzer(texts, strategy=WordCountStrategy.WORD_PUNKT).extract_dataset_metric()
        self.assertEqual({8: 1, 10: 1, 13: 1, 16: 1}, df_res.to_dict()['count'])

    def test_split_word_count(self):
        df_res = WordCountAnalyzer(texts, strategy=WordCountStrategy.PYTHON).extract_dataset_metric()
        self.assertEqual({5: 1, 7: 1, 8: 1, 9: 1}, df_res.to_dict()['count'])

    def test_nltk_base_word_count_en(self):
        df_res = WordCountAnalyzer(texts, strategy=WordCountStrategy.NLTK_BASE, language='english')\
            .extract_dataset_metric()
        self.assertEqual({8: 1, 9: 1, 10: 1, 16: 1}, df_res.to_dict()['count'])

    def test_nltk_base_word_count_de(self):
        # 'U.S.?' is treated differently with german nltk tokenize
        df_res = WordCountAnalyzer(texts, strategy=WordCountStrategy.NLTK_BASE, language='german')\
            .extract_dataset_metric()
        self.assertEqual({8: 1, 10: 2, 16: 1}, df_res.to_dict()['count'])

    def test_twitter_word_count(self):
        df_res = WordCountAnalyzer(texts, strategy=WordCountStrategy.NLTK_TWEET).extract_dataset_metric()
        self.assertEqual({8: 1, 10: 1, 12: 1, 13: 1}, df_res.to_dict()['count'])

    def test_nist_word_count(self):
        df_res = WordCountAnalyzer(texts, strategy=WordCountStrategy.NLTK_NIST).extract_dataset_metric()
        self.assertEqual({8: 1, 10: 1, 12: 1, 16: 1}, df_res.to_dict()['count'])

    def test_spacy_word_count_en(self):
        df_res = WordCountAnalyzer(texts, strategy=WordCountStrategy.SPACY, language='english')\
            .extract_dataset_metric()
        self.assertEqual({8: 2, 11: 1, 14: 1}, df_res.to_dict()['count'])

    def test_spacy_word_count_de(self):
        df_res = WordCountAnalyzer(texts, strategy=WordCountStrategy.SPACY, language='german')\
            .extract_dataset_metric()
        self.assertEqual({8: 1, 9: 1, 10: 1, 14: 1}, df_res.to_dict()['count'])

    def test_textblob_word_count(self):
        df_res = WordCountAnalyzer(texts, strategy=WordCountStrategy.TEXTBLOB).extract_dataset_metric()
        self.assertEqual({7: 2, 8: 1, 10: 1}, df_res.to_dict()['count'])

    def test_gensim_word_count(self):
        df_res = WordCountAnalyzer(texts, strategy=WordCountStrategy.GENSIM).extract_dataset_metric()
        self.assertEqual({7: 2, 9: 1, 10: 1}, df_res.to_dict()['count'])