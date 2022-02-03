import unittest

from zimp.pos.countvectorizer_analyzer import CountVectorizerAnalyzer
from zimp.pos.tokenization.builder import TokenizerStrategy

texts = [
    'How many words are in this sentence?',
    'How do you handle compound-words in the U.S.?',
    'Ehm,do you handle missing spaces?No?',
    'You can also distinguish uppercase and lowercase, right?'
    '@username213: Twitter  vocabulary of @otheruser12 miGhta bedifferent,hu? xD <3'
]


class CountVectorizerAnalyzerTest(unittest.TestCase):
    """
    see this as a regression test :-)
    """

    def test_regex_vectorizer(self):
        df_res = CountVectorizerAnalyzer(texts, strategy=TokenizerStrategy.REGEX).extract_dataset_metric()
        self.assertEqual(
            {'you': 3, 'how': 2, 'words': 2, 'in': 2, 'do': 2, 'handle': 2, 's': 1, 'sentence': 1, 'spaces': 1,
             'the': 1, 'this': 1, 'twitter': 1, 'otheruser12': 1, 'u': 1, 'uppercase': 1, 'username213': 1,
             'vocabulary': 1, 'xd': 1, 'right': 1, '3': 1, 'of': 1, 'no': 1, 'also': 1, 'mighta': 1, 'many': 1,
             'lowercase': 1, 'hu': 1, 'ehm': 1, 'distinguish': 1, 'compound': 1, 'can': 1, 'bedifferent': 1, 'are': 1,
             'and': 1, 'missing': 1}, df_res.to_dict()['count'])

    def test_regex_vectorizer_with_casing(self):
        df_res = CountVectorizerAnalyzer(texts, strategy=TokenizerStrategy.REGEX, lowercase=False) \
            .extract_dataset_metric()
        self.assertEqual(
            {'in': 2, 'do': 2, 'words': 2, 'handle': 2, 'you': 2, 'How': 2, 'Twitter': 1, 'miGhta': 1, 'xD': 1,
             'vocabulary': 1, 'username213': 1, 'uppercase': 1, 'this': 1, 'the': 1, 'spaces': 1, 'sentence': 1,
             'right': 1, 'otheruser12': 1, 'of': 1, 'missing': 1, 'many': 1, 'U': 1, 'lowercase': 1, 'Ehm': 1, 'hu': 1,
             'No': 1, 'S': 1, 'distinguish': 1, 'compound': 1, 'can': 1, 'bedifferent': 1, 'are': 1, 'and': 1,
             'also': 1, 'You': 1, '3': 1}, df_res.to_dict()['count'])

    def test_python_vectorizer(self):
        df_res = CountVectorizerAnalyzer(texts, strategy=TokenizerStrategy.PYTHON).extract_dataset_metric()
        self.assertEqual(
            {'you': 3, 'in': 2, 'how': 2, 'handle': 2, 'the': 1, 'right?@username213:': 1, 'sentence?': 1,
             'spaces?no?': 1, 'this': 1, 'missing': 1, 'twitter': 1, 'u.s.?': 1, 'uppercase': 1, 'vocabulary': 1,
             'words': 1, 'xd': 1, 'of': 1, '<3': 1, 'mighta': 1, '@otheruser12': 1, 'lowercase,': 1, 'ehm,do': 1,
             'do': 1, 'distinguish': 1, 'compound-words': 1, 'can': 1, 'bedifferent,hu?': 1, 'are': 1, 'and': 1,
             'also': 1, 'many': 1}, df_res.to_dict()['count'])

    def test_nltk_base_vectorizer(self):
        df_res = CountVectorizerAnalyzer(texts, strategy=TokenizerStrategy.NLTK_BASE).extract_dataset_metric()
        self.assertEqual(
            {'?': 6, ',': 3, 'you': 3, 'do': 2, '@': 2, 'in': 2, 'how': 2, 'handle': 2, 'this': 1, 'right': 1,
             'sentence': 1, 'spaces': 1, 'the': 1, '<': 1, 'of': 1, 'twitter': 1, 'u.s.': 1, 'uppercase': 1,
             'username213': 1, 'vocabulary': 1, 'words': 1, 'xd': 1, 'otheruser12': 1, 'mighta': 1, 'no': 1,
             'missing': 1, ':': 1, 'many': 1, '3': 1, 'hu': 1, 'ehm': 1, 'distinguish': 1, 'compound-words': 1,
             'can': 1, 'bedifferent': 1, 'are': 1, 'and': 1, 'also': 1, 'lowercase': 1}, df_res.to_dict()['count'])

    def test_wordpunkt_vectorizer(self):
        df_res = CountVectorizerAnalyzer(texts, strategy=TokenizerStrategy.WORD_PUNKT).extract_dataset_metric()
        self.assertEqual(
            {'?': 4, ',': 3, 'you': 3, 'in': 2, 'words': 2, 'how': 2, 'handle': 2, 'do': 2, 'sentence': 1, 'no': 1,
             'of': 1, 'otheruser12': 1, 'right': 1, 's': 1, 'spaces': 1, 'mighta': 1, 'the': 1, 'this': 1, 'twitter': 1,
             'u': 1, 'uppercase': 1, 'username213': 1, 'vocabulary': 1, 'xd': 1, 'missing': 1, '.?': 1, 'many': 1,
             '3': 1, ':': 1, '<': 1, '.': 1, '?@': 1, '@': 1, 'also': 1, 'and': 1, 'are': 1, 'bedifferent': 1, 'can': 1,
             'compound': 1, 'distinguish': 1, 'ehm': 1, 'hu': 1, '-': 1, 'lowercase': 1}, df_res.to_dict()['count'])

    def test_nltk_base_vectorizer_de(self):
        df_res = CountVectorizerAnalyzer(texts, strategy=TokenizerStrategy.NLTK_BASE, language='german') \
            .extract_dataset_metric()
        # note u.s. is now split into 'u.s' and '.' (compare to english nltk base)
        self.assertEqual(
            {'?': 6, ',': 3, 'you': 3, 'do': 2, '@': 2, 'in': 2, 'how': 2, 'handle': 2, ':': 1, 'otheruser12': 1,
             'right': 1, 'sentence': 1, 'spaces': 1, 'the': 1, 'this': 1, 'no': 1, 'twitter': 1, 'u.s': 1,
             'uppercase': 1, 'username213': 1, 'vocabulary': 1, 'words': 1, 'xd': 1, 'of': 1, 'many': 1, 'missing': 1,
             'mighta': 1, '<': 1, '.': 1, 'hu': 1, 'ehm': 1, 'distinguish': 1, 'compound-words': 1, 'can': 1,
             'bedifferent': 1, 'are': 1, 'and': 1, 'also': 1, '3': 1, 'lowercase': 1}, df_res.to_dict()['count'])

    def test_twitter_vectorizer(self):
        df_res = CountVectorizerAnalyzer(texts, strategy=TokenizerStrategy.NLTK_TWEET).extract_dataset_metric()
        # twitter handles are preserved, emojis are preserved (<3)
        self.assertEqual(
            {'?': 6, ',': 3, 'you': 3, 'in': 2, '.': 2, 'how': 2, 'handle': 2, 'do': 2, 'missing': 1, 'of': 1,
             'right': 1, 's': 1, 'sentence': 1, 'spaces': 1, 'the': 1, 'this': 1, 'twitter': 1, 'u': 1, 'uppercase': 1,
             'vocabulary': 1, 'words': 1, 'xd': 1, 'no': 1, '<3': 1, 'mighta': 1, ':': 1, 'lowercase': 1, 'hu': 1,
             'ehm': 1, 'distinguish': 1, 'compound-words': 1, 'can': 1, 'bedifferent': 1, 'are': 1, 'and': 1, 'also': 1,
             '@username213': 1, '@otheruser12': 1, 'many': 1}, df_res.to_dict()['count'])

    def test_nist_vectorizer(self):
        df_res = CountVectorizerAnalyzer(texts, strategy=TokenizerStrategy.NLTK_NIST).extract_dataset_metric()
        self.assertEqual(
            {'?': 6, ',': 3, 'you': 3, 'do': 2, '@': 2, '.': 2, 'in': 2, 'how': 2, 'handle': 2, 'the': 1, 'right': 1,
             's': 1, 'sentence': 1, 'spaces': 1, ':': 1, 'of': 1, 'this': 1, 'twitter': 1, 'u': 1, 'uppercase': 1,
             'username213': 1, 'vocabulary': 1, 'words': 1, 'xd': 1, 'otheruser12': 1, 'mighta': 1, 'no': 1,
             'missing': 1, '<': 1, 'many': 1, 'hu': 1, 'ehm': 1, 'distinguish': 1, 'compound-words': 1, 'can': 1,
             'bedifferent': 1, 'are': 1, 'and': 1, 'also': 1, '3': 1, 'lowercase': 1}, df_res.to_dict()['count'])

    def test_spacy_vectorizer_en(self):
        df_res = CountVectorizerAnalyzer(texts, strategy=TokenizerStrategy.SPACY).extract_dataset_metric()
        self.assertEqual(
            {'?': 4, 'you': 3, ',': 3, 'in': 2, 'words': 2, 'how': 2, 'handle': 2, 'do': 2, 'sentence': 1, 'missing': 1,
             'of': 1, 'right?@username213': 1, 'spaces?no': 1, 'many': 1, 'the': 1, 'this': 1, 'twitter': 1, 'u.s': 1,
             'uppercase': 1, 'vocabulary': 1, 'xd': 1, 'mighta': 1, ' ': 1, 'lowercase': 1, 'ehm': 1, 'distinguish': 1,
             'compound': 1, 'can': 1, 'bedifferent': 1, 'are': 1, 'and': 1, 'also': 1, '@otheruser12': 1, '<3': 1,
             ':': 1, '.': 1, '-': 1, 'hu': 1}, df_res.to_dict()['count'])

    def test_spacy_vectorizer_de(self):
        df_res = CountVectorizerAnalyzer(texts, strategy=TokenizerStrategy.SPACY, language='german') \
            .extract_dataset_metric()
        # interestingly 'compound-words' is treated differently in comparison to the english spacy model
        self.assertEqual(
            {'?': 5, 'you': 3, ',': 3, 'do': 2, 'how': 2, 'handle': 2, 'in': 2, 'also': 1, 'missing': 1, 'xd': 1,
             'words': 1, 'vocabulary': 1, 'uppercase': 1, 'u.s': 1, 'twitter': 1, 'this': 1, 'the': 1, 'spaces': 1,
             'sentence': 1, 'right?@username213': 1, 'of': 1, 'no': 1, 'mighta': 1, 'and': 1, 'many': 1, 'lowercase': 1,
             '.': 1, 'hu': 1, ':': 1, '<3': 1, 'ehm': 1, '@otheruser12': 1, 'distinguish': 1, 'compound-words': 1,
             'can': 1, 'bedifferent': 1, 'are': 1, ' ': 1}, df_res.to_dict()['count'])

    def test_gensim_vectorizer(self):
        df_res = CountVectorizerAnalyzer(texts, strategy=TokenizerStrategy.GENSIM).extract_dataset_metric()
        self.assertEqual(
            {'you': 3, 'do': 2, 'in': 2, 'how': 2, 'handle': 2, 'words': 2, 'username': 1, 'uppercase': 1,
             'vocabulary': 1, 'u': 1, 'otheruser': 1, 'twitter': 1, 'xd': 1, 'this': 1, 'the': 1, 'spaces': 1,
             'sentence': 1, 's': 1, 'right': 1, 'also': 1, 'of': 1, 'and': 1, 'missing': 1, 'mighta': 1, 'many': 1,
             'lowercase': 1, 'hu': 1, 'ehm': 1, 'distinguish': 1, 'compound': 1, 'can': 1, 'bedifferent': 1, 'are': 1,
             'no': 1}, df_res.to_dict()['count'])
