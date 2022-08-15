import unittest

from zimp.readability.winnetka import PrepositionFrequencyScore, PosMethod, WinnetkaScore, SimpleSentenceFilter

sentence_a = 'What is the average speed of the horses at the Kentucky Derby?'
sentence_b = 'Who developed the vaccination against polio?'
sentence_c = 'What is the name of the chocolate company in San Francisco?'
sentence_d = 'This is an example for a non-simple sentence, but it might be a bad one.'


class WinnetkaScoreTest(unittest.TestCase):

    def test_prep_frequency_spacy(self):
        pfs = PrepositionFrequencyScore(pos_method=PosMethod.SPACY)
        self.assertEqual(153.84615384615387, pfs.get_score(sentence_a))

    def test_prep_frequeny_stanza(self):
        # same results for simple sentences - yeah!
        pfs = PrepositionFrequencyScore(pos_method=PosMethod.STANZA)
        self.assertEqual(153.84615384615387, pfs.get_score(sentence_a))

    def test_simple_sentence_filter(self):
        ssf = SimpleSentenceFilter()
        simple_sents, _ = ssf.get_simple_sentences(sentence_a)
        self.assertEqual(1, len(simple_sents))

    def test_nonsimple_sentence_filter(self):
        ssf = SimpleSentenceFilter()
        simple_sents, _ = ssf.get_simple_sentences(sentence_d)
        self.assertEqual(0, len(simple_sents))

    def test_nonsimple_de_sentence_filter(self):
        # two verbs
        ssf = SimpleSentenceFilter(language='german')
        simple_sents, _ = ssf.get_simple_sentences('Wer nach einem Zusammenhang sucht, findet ihn im Bekenntnis zur Abschweifung')
        self.assertEqual(0, len(simple_sents))

    def test_winnetka_sentence(self):
        ws = WinnetkaScore()
        self.assertEqual(213.4511538461539, ws.get_score(sentence_a))

    def test_winnetka_dataset(self):
        ws = WinnetkaScore()
        self.assertEqual(230.69875000000002, ws.get_dataset_score([sentence_a, sentence_b, sentence_c]))

