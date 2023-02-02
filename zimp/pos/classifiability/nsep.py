import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from zimp.pos.classifiability.base import DatasetScore
from zimp.pos.tokenization.builder import TokenizerStrategy, build_tokenizer
from sklearn.tree import DecisionTreeClassifier, _tree


class SeparabilityScore(DatasetScore):

    def __init__(self, min_freq=1, n_gram_words=1,
                 strategy: TokenizerStrategy = TokenizerStrategy.NLTK_BASE,
                 language: str = 'english', lowercase=True):
        """
        :param min_freq: minimum number of observations an n-gram needs to be seen to be considered
        :param n_gram_words: word engrams to be used as predictors
        :param strategy: strategy to tokenize text into words
        :param language: only used for nltk-base tokenizer and spacy, for custom rules
        :param lowercase: if true, all input text is transformed to lower case
        """
        self.min_freq = min_freq
        self.strategy = strategy
        self.language = language
        self.tokenizer = build_tokenizer(strategy, language)
        self.lowercase = lowercase
        self.count_vectorizer = CountVectorizer(
            tokenizer=self.tokenizer.tokenize_text,
            lowercase=lowercase,
            ngram_range=(n_gram_words, n_gram_words)
        )

    def score(self, X, y):
        y = np.array(y)
        X = np.array(X)

        X_c = self.count_vectorizer.fit_transform(X)
        # we only consider binary features (word present or not)
        X_c.data = np.ones(X_c.data.shape)

        clf = DecisionTreeClassifier(random_state=123)  # fully expanded tree
        clf.fit(X_c, y)
        tree = clf.tree_

        # count all wrongly split input samples
        leaf_node_idx = tree.feature == _tree.TREE_UNDEFINED
        incorrect_sample_cnt = tree.value[leaf_node_idx].min(axis=2)

        # ratio of correctly splittable samples
        return 1 - incorrect_sample_cnt.sum() / X.size



