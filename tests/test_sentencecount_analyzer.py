import unittest

from zimp.pos.sentence_count_analyzer import SentenceCountAnalyzer
from zimp.pos.tokenization.builder import SentenceTokenizerStrategy

texts = [
    "This is a simple sentence without punctuation",
    "Let's test how Mr. Analyzer handles abbreviations.",
    "Pay me $12.23 on 12.01.2012, please.",
    "What about this? We have a sentence. No, we have three.",
    "Uncommon abbreviations like F.B.C. exist outside the U.S. too.",
    "The band Wham! might prove difficult for our tokenizers.",
]


class SentenceCountAnalyzerTest(unittest.TestCase):

    def test_simple_sentence_count(self):
        sca = SentenceCountAnalyzer(texts, SentenceTokenizerStrategy.SIMPLE)
        self.assertEqual([1, 2, 4, 3, 6, 2], sca.extract_batch_metrics())

    def test_punkt_sentence_count(self):
        # fails for hard examples
        sca = SentenceCountAnalyzer(texts, SentenceTokenizerStrategy.PUNKT)
        self.assertEqual([1, 1, 1, 3, 2, 2], sca.extract_batch_metrics())

    def test_corpus_punkt_sentence_count(self):
        # fails for wham!
        sca = SentenceCountAnalyzer(texts, SentenceTokenizerStrategy.CORPUS_PUNKT)
        self.assertEqual([1, 1, 1, 3, 1, 2], sca.extract_batch_metrics())

    def test_europarl_sentence_count(self):
        # works for our examples :-)
        sca = SentenceCountAnalyzer(texts, SentenceTokenizerStrategy.EUROPARL)
        self.assertEqual([1, 1, 1, 3, 1, 1], sca.extract_batch_metrics())

    def test_spacy_sentence_count(self):
        # fails for wham!
        sca = SentenceCountAnalyzer(texts, SentenceTokenizerStrategy.SPACY)
        self.assertEqual([1, 1, 1, 3, 1, 2], sca.extract_batch_metrics())
