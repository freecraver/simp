from enum import Enum

import spacy.tokens

from zimp.pos.countvectorizer_analyzer import CountVectorizerAnalyzer
from zimp.pos.spacy import get_or_download_model
from zimp.pos.stanza import get_or_download_model as get_or_download_stanza_model
from zimp.pos.tokenization.builder import TokenizerStrategy
from zimp.readability.lively import OutOfVocabularySizeScore
from zimp.readability.metrics import ReadabilityScore
from zimp.readability.util import get_freq_score

"""
Vogel, Mabel, and Carleton Washburne.
"An objective method of determining grade placement of children's reading material."
The Elementary School Journal 28.5 (1928): 373-381.
"""


class PosMethod(Enum):
    SPACY = 0
    STANZA = 1


class WinnetkaScore(ReadabilityScore):
    """
    higher score means higher difficulty; reference:
    Score........................................... Grade
    4–16............................................ II
    18–34........................................... III
    36–52........................................... IV
    54–62........................................... V
    64–70........................................... VI
    72–78........................................... VII
    80–86........................................... VIII
    88–94........................................... IX
    96–102.......................................... X
    104–112......................................... XI
    """

    def __init__(
            self,
            language='english',
            tokenizer_strategy: TokenizerStrategy = TokenizerStrategy.NLTK_BASE,
            pos_method: PosMethod = PosMethod.SPACY,
            s=1000
    ):
        """
        :param language:
        :param tokenizer_strategy: strategy used to tokenize words
        :param pos_method: method for pos-tag extraction
        :param s: scaling factor - inner metrics are normalized per s words
        """
        self.s = s
        self.tokenizer_strategy = tokenizer_strategy
        self.language = language
        self.prep_extractor = self.prep_extractor = build_preposition_extractor(pos_method, language)
        self.oov_score = OutOfVocabularySizeScore(word_tokenizer_strategy=tokenizer_strategy, language=language)
        self.simple_sentence_filter = SimpleSentenceFilter(language=language)

    def get_score(self, text: str) -> float:
        return self.get_dataset_score([text])

    def get_dataset_score(self, texts) -> float:
        cva = CountVectorizerAnalyzer(texts, self.tokenizer_strategy, self.language)
        word_counts = cva.extract_dataset_metric()
        X_2 = word_counts.index.size / word_counts.sum(axis=0).iloc[0] * self.s
        X_3 = get_freq_score(texts, self.prep_extractor.get_prepositions) * self.s
        X_4 = get_freq_score(texts, self.oov_score.get_filtered_toks) * self.s
        X_5 = get_freq_score(texts, self.simple_sentence_filter.get_simple_sentences) * 75  # fixed to base 75, see Winnetka paper

        return 0.085 * X_2 + 0.101 * X_3 + 0.604 * X_4 - 0.411 * X_5 + 17.43


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
        self.prep_extractor = build_preposition_extractor(pos_method, language)
        self.s = s

    def get_score(self, text: str) -> float:
        preps, toks = self.prep_extractor.get_prepositions(text)
        prep_ratio = len(preps) / len(toks)
        return prep_ratio * self.s


class SimpleSentenceFilter:
    """
    returns sentences without dependent or co-ordinate clauses
    approximates this by (only) checking for conjunctions-dependency tags
    """

    def __init__(self, language: str = 'english'):
        self.base_model = get_or_download_model(language)
        self.complex_sentence_pos_indicators = [
            'SCONJ',  # subordinating conjunction, e.g. "I believe _that_ he will come"
        ]
        # dependency tags are different for german parser
        self.subject_dep_indicators = ['sb'] if language == 'german' else ['nsubj']

    def get_simple_sentences(self, text):
        sentences = list(self.base_model(text).sents)
        simple_sentences = [s for s in sentences if self.is_simple_sentence(s)]
        return simple_sentences, sentences

    def is_simple_sentence(self, sentence: spacy.tokens.Span):
        complex_tokens = []
        verb_tokens = []
        subj_tokens = []

        for idx, t in enumerate(sentence):
            if t.pos_ in self.complex_sentence_pos_indicators and idx > 0:
                # sconj at beginning of the sentence are not an indicator for a non-simple sentence (german)
                complex_tokens.append(t)
            if t.pos_ == 'VERB':
                verb_tokens.append(t)
            if t.dep_ in self.subject_dep_indicators:
                subj_tokens.append(t)

        # no complex sentence indicators, not more than one subject and one verb
        return len(complex_tokens) < 1 and len(verb_tokens) < 2 and len(subj_tokens) < 2


def build_preposition_extractor(pos_method: PosMethod, language: str):
    return _SpacyPrepositionExtractor(language) if pos_method == PosMethod.SPACY else \
            _StanzaPrepositionExtractor(language)


class _SpacyPrepositionExtractor:
    """
    the german spacy model does not use UD tags, and the english one includes multiple preposition definitions (prep,
    agent) - for a clear definition I decided to stick with adpositions (this includes postpositions)
    """

    def __init__(self, language='english'):
        self.base_model = get_or_download_model(language)

    def get_prepositions(self, text):
        toks = self.base_model(text)
        return [t for t in toks if t.pos_ == 'ADP'], toks


class _StanzaPrepositionExtractor:
    """
    stanza does not differ between pre- and postpositions, so we need to use adpositions (ie both)
    """

    def __init__(self, language='english'):
        self.base_model = get_or_download_stanza_model(language, 'tokenize, pos')

    def get_prepositions(self, text):
        doc = self.base_model(text)
        toks = [w for s in doc.sentences for w in s.words]
        return [t for t in toks if t.upos == 'ADP'], toks

