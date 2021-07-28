import re

EMOJI_REGEX = r'(\u00a9|\u00ae|[\u2000-\u3300]|\ud83c[\ud000-\udfff]|\ud83d[\ud000-\udfff]|\ud83e[\ud000-\udfff])'
HASHTAG_REGEX = r'\B#([a-zA-Z]+\b)'
URL_REGEX = r'(\b(https?|ftp|file)://)[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]'


def get_all(post, regex_pattern):
    """
    Returns all strings that match a prescribed regex pattern

    Parameters
    ----------
    post : UniMessage
    regex_pattern : str

    Returns
    -------
    list(str)
        The extracted string pattern
    """
    return [x.group() for x in re.finditer(regex_pattern, post.text)]
