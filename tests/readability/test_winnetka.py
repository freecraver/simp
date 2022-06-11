import unittest

from zimp.readability.winnetka import PrepositionFrequencyScore, PosMethod

sentence_a = 'What is the average speed of the horses at the Kentucky Derby?'


class WinnetkaScoreTest(unittest.TestCase):

    def test_prep_frequency_spacy(self):
        pfs = PrepositionFrequencyScore(pos_method=PosMethod.SPACY)
        self.assertEqual(153.84615384615387, pfs.get_score(sentence_a))

    def test_prep_frequeny_stanza(self):
        # same results for simple sentences - yeah!
        pfs = PrepositionFrequencyScore(pos_method=PosMethod.STANZA)
        self.assertEqual(153.84615384615387, pfs.get_score(sentence_a))