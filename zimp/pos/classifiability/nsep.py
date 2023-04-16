import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from zimp.pos.classifiability.base import DatasetScore
from zimp.pos.tokenization.builder import TokenizerStrategy, build_tokenizer
from sklearn.tree import DecisionTreeClassifier, _tree
from dataclasses import dataclass


@dataclass
class SeparabilityScoreDetails:
    split_score: float  # ratio of correctly splittable samples
    max_feature_importance: float  # highest importance of a single n-gram, ie total reduction of the trained criterion brought by it
    used_features: int  # number of n-grams used for prediction
    tree_depth: int  # depth of the trained tree

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
        self._last_clf = None

    def score(self, X, y):
        y = np.array(y)
        X = np.array(X)

        X_c = self.count_vectorizer.fit_transform(X)
        # we only consider binary features (word present or not)
        X_c.data = np.ones(X_c.data.shape)

        clf = DecisionTreeClassifier(random_state=123)  # fully expanded tree
        clf.fit(X_c, y)
        self._last_clf = clf
        tree = clf.tree_

        # count all wrongly split input samples
        leaf_node_idx = tree.feature == _tree.TREE_UNDEFINED
        incorrect_sample_cnt = tree.value[leaf_node_idx].min(axis=2)

        # ratio of correctly splittable samples
        return 1 - incorrect_sample_cnt.sum() / X.size

    def score_detailed(self, X, y) -> SeparabilityScoreDetails:
        split_score = self.score(X, y)
        tree = self._last_clf.tree_
        feature_importances = self._last_clf.feature_importances_
        return SeparabilityScoreDetails(
            split_score=split_score,
            max_feature_importance=feature_importances.max(),
            used_features=np.count_nonzero(self._last_clf.feature_importances_),
            tree_depth=tree.max_depth
        )


