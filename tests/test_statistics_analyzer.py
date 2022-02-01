import unittest

from zimp.pos.statistics_analyzer import TextLengthAnalyzer


class StatisticsAnalyzerTest(unittest.TestCase):

    def test_text_length(self):
        tla = TextLengthAnalyzer(['How many characters are in this sentence?'])
        df_res = tla.extract_dataset_metric()
        self.assertEqual((1, 1), df_res.shape)
        self.assertEqual(41, df_res.index[0])
        self.assertEqual(1, df_res['count'][41])

    def test_text_length_mutliple(self):
        tla = TextLengthAnalyzer(['How many characters are in this sentence?', 'And what about this one?'])
        df_res = tla.extract_dataset_metric()
        self.assertEqual(1, df_res['count'][41])
        self.assertEqual(1, df_res['count'][24])


if __name__ == '__main__':
    unittest.main()
