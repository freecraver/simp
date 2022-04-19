import unittest

from zimp.readability.vocab import build_word_vocab, build_word_support

sentences = [
    'Thigh-high in the water, the fisherman’s hope for dinner soon turned to despair.',
    'The sun had set and so had his dreams.',
    'On each full moon',
    'There can never be too many cherries on an ice cream sundae.',
    'There are over 500 starfish in the bathroom drawer.'
]


class VocabUtilsTest(unittest.TestCase):

    def test_build_vocab(self):
        vocab = build_word_vocab(sentences)
        self.assertEqual(
            ['.', 'the', 'had', 'on', 'there', 'in', 'so', 'many', 'moon', 'never', 'over', 's', 'set', ',', 'soon',
             'sun', 'sundae', 'thigh-high', 'to', 'too', 'turned', 'water', 'starfish', 'ice', 'hope', 'his', '500',
             'an', 'and', 'are', 'bathroom', 'be', 'can', 'cherries', 'cream', 'despair', 'dinner', 'drawer', 'dreams',
             'each', 'fisherman', 'for', 'full', '’'],
            vocab
        )

    def test_build_restricted_vocab(self):
        vocab = build_word_vocab(sentences, max_n=5)
        self.assertEqual(['.', 'the', 'had', 'on', 'there'], vocab)

    def test_build_word_support(self):
        support_dict = build_word_support(sentences)
        self.assertEqual(0, support_dict['supercalifragilisticexpialidocious'])
        self.assertEqual(43, support_dict['the'])

    def test_build_restricted_word_support(self):
        support_dict = build_word_support(sentences, max_n=5)
        self.assertEqual(0, support_dict['supercalifragilisticexpialidocious'])
        self.assertEqual(4, support_dict['the'])
