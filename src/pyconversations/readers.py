from abc import ABC, abstractmethod


class RawReader(ABC):
    """
    A base class for reading of raw files.

    To cache the pre-processing this package does,
    one will typically:
    * Use a raw reader
    * Cache conversationally-scoped chunks in the UniversalPost format
    """

    def __init__(self):
        pass

    @abstractmethod
    def read(self):
        pass
