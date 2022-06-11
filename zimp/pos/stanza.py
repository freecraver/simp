import stanza


def get_or_download_model(language, processors={}) -> stanza.pipeline:
    """
    :param language: english or german
    :param processors: specific pipeline processors, see stanza doc
    :return:
    """
    model_name = _get_model_name(language)
    stanza.download(model_name)
    return stanza.Pipeline(model_name, processors=processors)


def _get_model_name(language):
    language_prefix = 'en'
    if language == 'german':
        language_prefix = 'de'
    elif language != 'english':
        raise ValueError(f'Language {language} not yet supported')

    return language_prefix