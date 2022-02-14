from enum import Enum

from zimp.simplification.simplifier import BaseSimplifier
from zimp.simplification.symbol import SimpleCharacterSimplifier, VocabularyCharacterSimplifier


class SimplificationStrategy(Enum):

    SYMBOL_SIMPLE = "SYMBOL_SIMPLE"
    SYMBOL_SIMPLE_ALPHA_NOCASING = "SYMBOL_SIMPLE_ALPHA_NOCASING"
    SYMBOL_VOCAB = "SYMBOL_VOCAB"
    SYMBOL_VOCAB_NOCASING = "SYMBOL_VOCAB_NOCASING"


def build_simplifier(strategy: SimplificationStrategy, **kwargs) -> BaseSimplifier:
    if strategy == SimplificationStrategy.SYMBOL_SIMPLE:
        return SimpleCharacterSimplifier(**kwargs)
    elif strategy == SimplificationStrategy.SYMBOL_SIMPLE_ALPHA_NOCASING:
        return SimpleCharacterSimplifier(lowercase=True, only_alphabetic=True, **kwargs)
    elif strategy == SimplificationStrategy.SYMBOL_VOCAB:
        return VocabularyCharacterSimplifier(lowercase=False, **kwargs)
    elif strategy == SimplificationStrategy.SYMBOL_VOCAB_NOCASING:
        return VocabularyCharacterSimplifier(lowercase=True, **kwargs)
