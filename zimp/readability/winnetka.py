from enum import Enum

from zimp.pos.spacy import get_or_download_model
from zimp.pos.stanza import get_or_download_model as get_or_download_stanza_model
from zimp.readability.metrics import ReadabilityScore

"""
Vogel, Mabel, and Carleton Washburne.
"An objective method of determining grade placement of children's reading material."
The Elementary School Journal 28.5 (1928): 373-381.
"""


class PosMethod(Enum):
    SPACY = 0
    STANZA = 1


class PrepositionFrequencyScore(ReadabilityScore):
    """
    returns the number of prepositions (in, of,..) per s words

    """

    def __init__(self, language='english', pos_method=PosMethod.SPACY, s=1000):
        """
        :param language:
        :param pos_method: method used for part of speech tagging
        :param s: scaling factor s (-> number of prepositions per s words)
        """
        self.pos_method = pos_method
        self.prep_extractor = _SpacyPrepositionExtractor(language) if pos_method == PosMethod.SPACY else \
            _StanzaPrepositionExtractor(language)
        self.s = s

    def get_score(self, text: str) -> float:
        prep_ratio = self.prep_extractor.get_preposition_ratio(text)
        return prep_ratio * self.s


class _SpacyPrepositionExtractor:

    def __init__(self, language='english'):
        self.base_model = get_or_download_model('core_web_sm', language)

    def get_preposition_ratio(self, text):
        toks = self.base_model(text)
        return len([t for t in toks if t.dep_ == 'prep']) / len(toks)


class _StanzaPrepositionExtractor:
    """
    stanza does not differ between pre- and postpositions, so we need to use adpositions (ie both)
    """

    def __init__(self, language='english'):
        self.base_model = get_or_download_stanza_model(language, 'tokenize, pos')

    def get_preposition_ratio(self, text):
        doc = self.base_model(text)
        toks = [w for s in doc.sentences for w in s.words]
        return len([t for t in toks if t.upos == 'ADP']) / len(toks)

