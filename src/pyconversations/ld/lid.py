from langid.langid import LanguageIdentifier
from langid.langid import model

from .base import BaseLangDetect


class LangidLangDetect(BaseLangDetect):

    """
    Language detection using the langid package
    """

    def __init__(self):
        self._model = LanguageIdentifier.from_modelstring(model, norm_probs=True)

    def get(self, text):
        """
        Uses langid module to detect a language

        Parameters
        ----------
        text : str
            The raw text to detect the language of

        Returns
        -------
        tuple(str, float)
            The detected language and confidence of detection
        """

        lang, conf = self._model.classify(text)

        return lang, conf
