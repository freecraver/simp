from enum import Enum

from zimp.simplification.simplifier import BaseSimplifier
from zimp.simplification.symbol import SimpleCharacterSimplifier, VocabularyCharacterSimplifier
from zimp.simplification.word import SimpleWordSimplifier, VocabularyWordSimplifier


class SimplificationStrategy(Enum):

    SYMBOL_SIMPLE = "SYMBOL_SIMPLE"
    SYMBOL_SIMPLE_ALPHA_NOCASING = "SYMBOL_SIMPLE_ALPHA_NOCASING"
    SYMBOL_VOCAB = "SYMBOL_VOCAB"
    SYMBOL_VOCAB_NOCASING = "SYMBOL_VOCAB_NOCASING"
    WORD_SIMPLE = "WORD_SIMPLE"
    WORD_VOCAB = "WORD_VOCAB"


def build_simplifier(strategy: SimplificationStrategy, **kwargs) -> BaseSimplifier:
    if strategy == SimplificationStrategy.SYMBOL_SIMPLE:
        return SimpleCharacterSimplifier(**kwargs)
    elif strategy == SimplificationStrategy.SYMBOL_SIMPLE_ALPHA_NOCASING:
        return SimpleCharacterSimplifier(lowercase=True, only_alphabetic=True, **kwargs)
    elif strategy == SimplificationStrategy.SYMBOL_VOCAB:
        return VocabularyCharacterSimplifier(lowercase=False, **kwargs)
    elif strategy == SimplificationStrategy.SYMBOL_VOCAB_NOCASING:
        return VocabularyCharacterSimplifier(lowercase=True, **kwargs)
    elif strategy == SimplificationStrategy.WORD_SIMPLE:
        return SimpleWordSimplifier(**kwargs)
    elif strategy == SimplificationStrategy.WORD_VOCAB:
        return VocabularyWordSimplifier(**kwargs)
