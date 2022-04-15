import unittest

from zimp.readability.metrics import SentenceLengthScore

sentence_a = 'This is a sentence. We treat punctuation as words. Is that really good?'


class SentenceLengthScoreTest(unittest.TestCase):

    def test_single_text(self):
        sls = SentenceLengthScore()
        self.assertEqual(5 + 1/3, sls.get_score(sentence_a))

    def test_multi_text(self):
        sls = SentenceLengthScore()
        self.assertEqual([3.0, 5.0], sls.get_scores(['This is short', 'This is a bit longer']))
