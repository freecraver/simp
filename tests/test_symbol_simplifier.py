import unittest

from zimp.simplification.builder import build_simplifier, SimplificationStrategy
from zimp.simplification.symbol import SimpleCharacterSimplifier, VocabularyCharacterSimplifier

texts = [
    'How many words are in this sentence?',
    'How do you handle compound-words in the U.S.?',
    'Ehm,do you handle missing spaces?No?',
    '@username213: Twitter  vocabulary of @otheruser12 miGhta bedifferent,hu? xD <3',
    "Why don't you handle clitics with apostrophes differently?"
]


class SymbolSimplifierTest(unittest.TestCase):

    def test_simplifier_casing(self):
        # only duplicate whitespace is removed
        s = build_simplifier(SimplificationStrategy.SYMBOL_SIMPLE, lowercase=False)
        self.assertEqual([
            'How many words are in this sentence?',
            'How do you handle compound-words in the U.S.?',
            'Ehm,do you handle missing spaces?No?',
            '@username213: Twitter vocabulary of @otheruser12 miGhta bedifferent,hu? xD <3',
            "Why don't you handle clitics with apostrophes differently?"
        ], s.simplify_dataset(texts))

    def test_simplifier_no_casing(self):
        s = build_simplifier(SimplificationStrategy.SYMBOL_SIMPLE)
        self.assertEqual([
            'how many words are in this sentence?',
            'how do you handle compound-words in the u.s.?',
            'ehm,do you handle missing spaces?no?',
            '@username213: twitter vocabulary of @otheruser12 mighta bedifferent,hu? xd <3',
            "why don't you handle clitics with apostrophes differently?"
        ], s.simplify_dataset(texts))

    def test_simplifier_alpha(self):
        s = build_simplifier(SimplificationStrategy.SYMBOL_SIMPLE_ALPHA_NOCASING)
        self.assertEqual([
            'how many words are in this sentence',
            'how do you handle compound words in the u s',
            'ehm do you handle missing spaces no',
            'username twitter vocabulary of otheruser mighta bedifferent hu xd',
            'why don t you handle clitics with apostrophes differently'
        ], s.simplify_dataset(texts))

    def test_simplifier_keep_whitespace(self):
        s = SimpleCharacterSimplifier(remove_duplicate_whitespace=False)
        self.assertEqual([
            'how many words are in this sentence?',
            'how do you handle compound-words in the u.s.?',
            'ehm,do you handle missing spaces?no?',
            '@username213: twitter  vocabulary of @otheruser12 mighta bedifferent,hu? xd <3',
            "why don't you handle clitics with apostrophes differently?"
        ], s.simplify_dataset(texts))

    def test_vocab_simplifier_tf_1(self):
        s = VocabularyCharacterSimplifier(texts)
        self.assertEqual([
            'How many words are in this sentence?',
            'How do you handle compound-words in the U.S.?',
            'Ehm,do you handle missing spaces?No?',
            '@username213: Twitter vocabulary of @otheruser12 miGhta bedifferent,hu? xD <3',
            "Why don't you handle clitics with apostrophes differently?"
        ], s.simplify_dataset(texts))

    def test_vocab_simplifier_tf_0d05(self):
        s = VocabularyCharacterSimplifier(texts, min_term_frequency=0.01, lowercase=True)
        self.assertEqual([
            'how many words are in this sentence?',
            'how do you handle compound words in the u.s.?',
            'ehm,do you handle missing spaces?no?',
            '@username213 twitter ocabulary of @otheruser12 mighta bedifferent,hu? d 3',
            'why don t you handle clitics with apostrophes differently?'
        ], s.simplify_dataset(texts))

    def test_vocab_simplifier_df_0d2(self):
        # 0.2 of 5 docs = 1 -> keep all which occur at least once
        s = VocabularyCharacterSimplifier(texts, min_document_frequency=0.2)
        self.assertEqual([
            'How many words are in this sentence?',
            'How do you handle compound-words in the U.S.?',
            'Ehm,do you handle missing spaces?No?',
            '@username213: Twitter vocabulary of @otheruser12 miGhta bedifferent,hu? xD <3',
            "Why don't you handle clitics with apostrophes differently?"
        ], s.simplify_dataset(texts))

    def test_vocab_simplifier_tf_4(self):
        # only keep character which occur at least four times
        s = VocabularyCharacterSimplifier(texts, min_term_frequency=4, lowercase=True)
        self.assertEqual([
            'how many words are in this sentence?',
            'how do you handle compound words in the u s ?',
            'ehm do you handle missin spaces?no?',
            'username twitter oca ulary of otheruser mi hta edifferent hu? d',
            'why don t you handle clitics with apostrophes differently?'
        ], s.simplify_dataset(texts))

    def test_uninitialized_vocab_simplifier(self):
        s = build_simplifier(SimplificationStrategy.SYMBOL_VOCAB_NOCASING, min_term_frequency=4)
        self.assertEqual([''], s.simplify_dataset(['No initialization - all gone']))

    def test_postinit_vocab_simplifier(self):
        s = build_simplifier(SimplificationStrategy.SYMBOL_VOCAB_NOCASING, min_term_frequency=4)
        s.load_parameters(dataset=texts)
        self.assertEqual([
            'how many words are in this sentence?',
            'how do you handle compound words in the u s ?',
            'ehm do you handle missin spaces?no?',
            'username twitter oca ulary of otheruser mi hta edifferent hu? d',
            'why don t you handle clitics with apostrophes differently?'
        ], s.simplify_dataset(texts))

