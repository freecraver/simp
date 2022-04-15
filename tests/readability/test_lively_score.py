import unittest

from zimp.readability.lively import OutOfVocabularySizeScore, VocabularySizeScore

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
