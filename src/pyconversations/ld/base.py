class BaseLangDetect:

    """
    Container for what a language detection module
    should take as input and should return as output
    """

    def get(self, text):
        """
        Uses the language detection module to detect a language

        Parameters
        ----------
        text : str
            The raw text to detect the language of

        Returns
        -------
        tuple(str, float)
            The detected language and confidence of detection
        """
        return 'und', 0.0  # (lang_str, confidence)
