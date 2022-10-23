import unittest

from zimp.readability.dale_chall import DaleChallScore

sentence_a = 'This is a sentence. We treat punctuation as words. Is that really good?'

class DaleChallTest(unittest.TestCase):

    def test_english_sentence(self):
        dcs = DaleChallScore()
        self.assertEqual(11.216208333333334, dcs.get_score(sentence_a))

    def test_german_sentence(self):
        dcs = DaleChallScore(language='german')
        self.assertEqual(13.536, dcs.get_score('Wer nach einem Zusammenhang sucht, findet ihn im Bekenntnis zur Abschweifung'))
