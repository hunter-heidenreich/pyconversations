import fasttext

from .base import BaseLangDetect


class FTLangDetect(BaseLangDetect):

    def __init__(self, path_to_pretrained_model='other/lid.176.bin'):
        self._model = fasttext.load_model(path_to_pretrained_model)

    def get(self, text):
        lang, conf = self._model.predict([text])

        return lang[0][0].replace('__label__', ''), conf[0][0]
