from abc import ABC
from abc import abstractmethod


class BaseLangDetect(ABC):

    """
    Abstract container for what a language detection module
    should take as input and should return as output
    """

    @abstractmethod
    def get(self, _):
        return 'und', 0.0  # (lang_str, confidence)
