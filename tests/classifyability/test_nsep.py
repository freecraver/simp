import unittest

import numpy as np

from tests.dummy.datasets import simple_spam_small, shopping_list_small
from zimp.pos.classifiability.nsep import SeparabilityScore


class SeparabilityScoreTest(unittest.TestCase):

    def test_unigram_separability(self):
        ds = simple_spam_small
        ses = SeparabilityScore(n_gram_words=1)
        self.assertEqual(1, ses.score(ds['X'], ds['y']))

    def test_bigram_separability(self):
        ds = simple_spam_small
        ses = SeparabilityScore(n_gram_words=2)
        self.assertEqual(1, ses.score(ds['X'], ds['y']))

    def test_unigram_nonseparability(self):
        ds = shopping_list_small
        ses = SeparabilityScore(n_gram_words=1)
        self.assertEqual(0.75, ses.score(ds['X'], ds['y']))

    def test_bigram_shopping_separability(self):
        ds = shopping_list_small
        ses = SeparabilityScore(n_gram_words=2)
        self.assertEqual(1, ses.score(ds['X'], ds['y']))

    def test_detailed_scoring(self):
        ds = shopping_list_small
        ses = SeparabilityScore()
        score_details = ses.score_detailed(ds['X'], ds['y'])
        self.assertTrue(np.isclose(2/3, score_details.max_feature_importance))
        self.assertEqual(0.75, score_details.split_score)
        self.assertEqual(2, score_details.tree_depth)
        self.assertEqual(2, score_details.used_features)

