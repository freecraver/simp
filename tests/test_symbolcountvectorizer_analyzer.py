import unittest

from zimp.pos.countvectorizer_analyzer import SymbolCountVectorizerAnalyzer

texts = [
    'How many words are in this sentence?',
    '@username213: Twitter  vocabulary of @otheruser12 miGhta bedifferent,hu? xD <3',
    'We also want to handle \t tabs',
]


class SymbolCountVectorizerAnalyzerTest(unittest.TestCase):
    """
    see this as a regression test :-)
    """

    def test_count_symbols(self):
        scva = SymbolCountVectorizerAnalyzer(texts)
        self.assertEqual(
            {'e': 14, 't': 10, 'a': 10, 'r': 8, 'n': 8, 's': 7, 'o': 7, 'i': 5, 'h': 5, 'w': 4, 'u': 4, 'd': 3, 'm': 3,
             'l': 3, 'b': 3, 'f': 3, 'y': 2, '1': 2, 'c': 2, '@': 2, '?': 2, '3': 2, '2': 2, 'W': 1, 'T': 1, 'H': 1,
             'G': 1, 'D': 1, '<': 1, ':': 1, 'v': 1, 'x': 1, ',': 1}, scva.extract_dataset_metric().to_dict()['count'])

    def test_count_symbols_preserve_whitespace(self):
        scva = SymbolCountVectorizerAnalyzer(texts, remove_whitespace=False)
        self.assertEqual(
            {' ': 21, 'e': 14, 't': 10, 'a': 10, 'r': 8, 'n': 8, 's': 7, 'o': 7, 'i': 5, 'h': 5, 'w': 4, 'u': 4, 'b': 3,
             'm': 3, 'f': 3, 'l': 3, 'd': 3, 'c': 2, 'y': 2, '@': 2, '?': 2, '3': 2, '2': 2, '1': 2, 'W': 1, 'T': 1,
             'H': 1, 'G': 1, 'D': 1, '<': 1, ':': 1, 'v': 1, ',': 1, 'x': 1, '\t': 1},
            scva.extract_dataset_metric().to_dict()['count'])

    def test_count_symbols_alpha(self):
        scva = SymbolCountVectorizerAnalyzer(texts, only_alphabetic=True)
        self.assertEqual(
            {'e': 14, 't': 10, 'a': 10, 'r': 8, 'n': 8, 'o': 7, 's': 7, 'i': 5, 'h': 5, 'w': 4, 'u': 4, 'b': 3, 'd': 3,
             'f': 3, 'l': 3, 'm': 3, 'y': 2, 'c': 2, 'G': 1, 'W': 1, 'T': 1, 'v': 1, 'H': 1, 'x': 1, 'D': 1},
            scva.extract_dataset_metric().to_dict()['count'])

    def test_count_symbols_alpha_lowercase(self):
        scva = SymbolCountVectorizerAnalyzer(texts, only_alphabetic=True, lowercase=True)
        self.assertEqual(
            {'e': 14, 't': 11, 'a': 10, 'r': 8, 'n': 8, 'o': 7, 's': 7, 'h': 6, 'w': 5, 'i': 5, 'd': 4, 'u': 4, 'm': 3,
             'b': 3, 'l': 3, 'f': 3, 'c': 2, 'y': 2, 'g': 1, 'v': 1, 'x': 1},
            scva.extract_dataset_metric().to_dict()['count'])
