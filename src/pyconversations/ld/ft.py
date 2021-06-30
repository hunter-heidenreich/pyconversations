import fasttext

from .base import BaseLangDetect


class FTLangDetect(BaseLangDetect):

    """
    FastText module for language detection.
    """

    def __init__(self, path_to_pretrained_model='other/lid.176.bin'):
        self._model = fasttext.load_model(path_to_pretrained_model)

    def get(self, text):
        """
        Uses FastText module to detect a language

        Parameters
        ----------
        text : str
            The raw text to detect the language of

        Returns
        -------
        tuple(str, float)
            The detected language and confidence of detection
        """

        lang, conf = self._model.predict([text])

        return lang[0][0].replace('__label__', ''), conf[0][0]
