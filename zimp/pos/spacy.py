from sys import stderr

import spacy


def get_or_download_model(model_name, language=None) -> spacy.pipeline:
    """
    :param model_name: e.g. 'core_web_sm', you may also include the language prefix as in 'en_core_web_sm'
    :param language: 'english' or 'german'
    :return:
    """
    model_name = _get_model_name(model_name, language)

    try:
        nlp = spacy.load(model_name)
    except OSError:
        print(f'Downloading spaCy model {model_name}', file=stderr)
        from spacy.cli import download
        download(model_name)
        nlp = spacy.load(model_name)

    return nlp


def _get_model_name(model_name, language=None):
    if not language:
        # requested absolute model name
        return model_name

    language_prefix = 'en'
    if language == 'german':
        language_prefix = 'de'
    elif language != 'english':
        raise ValueError(f'Language {language} not yet supported')

    return f'{language_prefix}_{model_name}'