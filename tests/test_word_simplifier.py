import unittest

from zimp.simplification.builder import build_simplifier, SimplificationStrategy

texts = [
    'How many words are in this sentence?',
    'How do you handle compound-words in the U.S.?',
    'Ehm,do you handle missing spaces?No?',
    'You can also distinguish uppercase and lowercase, right?',
    '@username213: Twitter  vocabulary of @otheruser12 miGhta bedifferent,hu? xD <3'
]


class WordSimplifierTest(unittest.TestCase):

    def test_simple_word_simplifier(self):
        s = build_simplifier(SimplificationStrategy.WORD_SIMPLE)
        self.assertEqual([
            'how many words are in this sentence ?',
            'how do you handle compound-words in the u.s. ?',
            'ehm , do you handle missing spaces ? no ?',
            'you can also distinguish uppercase and lowercase , right ?',
            '@ username213 : twitter vocabulary of @ otheruser12 mighta bedifferent , hu ? xd < 3'
        ], s.simplify_dataset(texts))

    def test_simple_word_simplifier_minmax_length(self):
        s = build_simplifier(SimplificationStrategy.WORD_SIMPLE, min_length=4, max_length=8)
        self.assertEqual([
            'many words this sentence',
            'handle u.s.',
            'handle missing spaces',
            'also right',
            'twitter mighta'
        ], s.simplify_dataset(texts))

    def test_vocab_simplifier_tf_1(self):
        s = build_simplifier(SimplificationStrategy.WORD_VOCAB, dataset=texts)
        self.assertEqual([
            'How many words are in this sentence ?',
            'How do you handle compound-words in the U.S. ?',
            'Ehm , do you handle missing spaces ? No ?',
            'You can also distinguish uppercase and lowercase , right ?',
            '@ username213 : Twitter vocabulary of @ otheruser12 miGhta bedifferent , hu ? xD < 3'
        ], s.simplify_dataset(texts))

    def test_vocab_simplifier_tf_2(self):
        s = build_simplifier(SimplificationStrategy.WORD_VOCAB, dataset=texts, min_term_frequency=2)
        self.assertEqual([
            'How in ?',
            'How do you handle in ?',
            ', do you handle ? ?',
            ', ?',
            '@ @ , ?'
        ], s.simplify_dataset(texts))

    def test_vocab_simplifier_df_025(self):
        s = build_simplifier(SimplificationStrategy.WORD_VOCAB, dataset=texts, max_document_frequency=.25)
        self.assertEqual([
            'many words are this sentence',
            'compound-words the U.S.',
            'Ehm missing spaces No',
            'You can also distinguish uppercase and lowercase right',
            'username213 : Twitter vocabulary of otheruser12 miGhta bedifferent hu xD < 3'
        ], s.simplify_dataset(texts))

    def test_vocab_simplifier_tf2_df_05(self):
        s = build_simplifier(SimplificationStrategy.WORD_VOCAB, dataset=texts, min_term_frequency=2,
                             max_document_frequency=.5, lowercase=True)
        self.assertEqual([
            'how in',
            'how do handle in',
            'do handle',
            '',
            '@ @'
        ], s.simplify_dataset(texts))


