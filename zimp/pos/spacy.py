from sys import stderr

import spacy


def get_or_download_model(language) -> spacy.pipeline:
    """
    :param language: 'english' or 'german'
    :return: default small model for the specific language
    """
    model_name = _get_model_name(language)

    try:
        nlp = spacy.load(model_name)
    except OSError:
        print(f'Downloading spaCy model {model_name}', file=stderr)
        from spacy.cli import download
        download(model_name)
        nlp = spacy.load(model_name)

    return nlp


def _get_model_name(language):
    if language == 'german':
        return 'de_core_news_sm'
    elif language != 'english':
        raise ValueError(f'Language {language} not yet supported')

    return 'en_core_web_sm'