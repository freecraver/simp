import unittest

from tests.readability.test_vocab import sentences
from zimp.readability.lively import OutOfVocabularySizeScore, VocabularySizeScore, LivelyScore

sentence_a = 'This is a sentence. We treat punctuation as words. Is that really good?'


class LivelyScoreTest(unittest.TestCase):

    def test_vocab_size(self):
        vss = VocabularySizeScore()
        self.assertEqual(14, vss.get_score(sentence_a))

    def test_oov_en_wordlist(self):
        oovss = OutOfVocabularySizeScore(language='english')
        self.assertEqual(4, oovss.get_score(sentence_a))

    def test_oov_de_wordlist(self):
        oovss = OutOfVocabularySizeScore(language='german')
        self.assertEqual(9, oovss.get_score(sentence_a))

    def test_oov_custom_wordlist(self):
        oovss = OutOfVocabularySizeScore(word_list=['.', '?', '!', 'is', 'a'])
        self.assertEqual(10, oovss.get_score(sentence_a))

    def test_lively_en_default(self):
        lsc = LivelyScore()
        self.assertEqual(4335.35, lsc.get_score(sentence_a))

    def test_lively_de_default(self):
        lsc = LivelyScore(language='german')
        self.assertEqual(1926.9999999999998, lsc.get_score(sentence_a))

    def test_lively_custom_wordlist(self):
        lsc = LivelyScore(word_list=['is', 'potato', 'tomato'])
        self.assertEqual(2, lsc.get_score('Is potato tomato'))
        self.assertEqual(7/3, lsc.get_score('Is potato potato'))
        self.assertEqual(5/3, lsc.get_score('Is tomato tomato'))
        self.assertEqual(1, lsc.get_score('Is tomato apple'))
        self.assertEqual(0, lsc.get_score('apples are green'))

    def test_lively_custom_dataset(self):
        lsc = LivelyScore(reference_texts=sentences, max_n=10)
        self.assertEqual(2/3, lsc.get_score(sentence_a))
