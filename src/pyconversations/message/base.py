import re
from abc import ABC
from abc import abstractmethod
from datetime import datetime

from ..ld import LangidLangDetect
from ..tokenizers import PartitionTokenizer

# Langauge detection module; do not initialize unless asked for!
DETECTOR = None


def get_detector():
    global DETECTOR
    if DETECTOR is None:
        # DETECTOR = gcld3.NNetLanguageIdentifier(min_num_bytes=0, max_num_bytes=1000)
        # DETECTOR = FTLangDetect()
        DETECTOR = LangidLangDetect()

    return DETECTOR


class UniMessage(ABC):

    """
    The Universal Message class.

    This is designed to be the abstract, baseline object
    that all social media posts / conversation turns
    inherit from.
    The only mandatory field is the uid, a unique field.
    """

    def __init__(self, uid,
                 text='', author=None,
                 created_at=None, reply_to=None, platform=None, lang=None, tags=None,
                 lang_detect=False, tokenizer=PartitionTokenizer):
        # a unique identifier
        self._uid = uid

        # the text of the post
        self._text = text

        # the username/name of the author
        self._author = author

        # created datetime object
        self._created_at = created_at

        # collection of IDs this post was generated in reply to
        self._reply_to = set() if not reply_to else set(reply_to)

        # any special tags or identifiers associated with this message
        self._tags = set() if not tags else set(tags)

        # platform name
        self._platform = platform

        # language
        self._lang = lang
        self._lang_detect = lang_detect
        self._detect_language()

        self._tok = tokenizer

    def __hash__(self):
        return self._uid

    def __repr__(self):
        return f'UniMessage({self._platform}::{self._author}::{self._created_at}::{self._text[:50]}::tags={",".join(self._tags)})'

    def __ior__(self, other):
        # Setting this to always take the larger text chunk...
        if len(self._text) < len(other.text):
            self._text = other.text

        if self._author is None:
            self._author = other.author

        if self._created_at is None:
            self._created_at = other.created_at
        elif self._created_at and other.created_at and other.created_at < self._created_at:
            self._created_at = other.created_at

        if self._lang is None:
            self._lang = other.lang

        self._reply_to |= other.reply_to
        self._tags |= other.tags

        return self

    def _detect_language(self):
        """
        Classifies the text of the post and updates the language field, if asked for.
        """
        if (not self._lang or self.lang == 'und') and self._lang_detect and self._text:
            res = get_detector().get(text=self.text)
            self.lang = res[0] if res[1] >= 0.5 else 'und'

    @staticmethod
    @abstractmethod
    def parse_raw(raw, lang_detect=False):
        """
        Abstract static method that must be implemented by all non-abstract child classes.
        Concrete implementations should specify how to parse the raw data into this object.

        Parameters
        ----------
        raw : JSON/dict
            The raw data to be pre-processed.
        lang_detect : bool
            A boolean which specifies whether language detection should be activated. (Default: False)
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def parse_datestr(x):
        """
        Abstract static method that specifies how to convert the native datetime string
        into a a Python datetime object.

        Parameters
        ----------
        x : str
            The raw datetime string
        """
        raise NotImplementedError

    @property
    def uid(self):
        """
        The unique identifier of this object.

        Returns
        -------
        UID
            Unique identifier for this message.
        """
        return self._uid

    @property
    def text(self):
        """
        The text associated with this message.

        Returns
        -------
        str
            Message text
        """
        return self._text

    @text.setter
    def text(self, t):
        """
        Updates the text field of this message.

        Parameters
        ----------
        t : str
            The new text

        Returns
        -------
        None
        """
        self._text = t
        self._lang = None
        self._detect_language()

    @property
    def author(self):
        """
        Returns the author of this message.

        Returns
        -------
        str
            Author name/username
        """
        return self._author

    @author.setter
    def author(self, a):
        """
        Updates the author of this message.

        Parameters
        ----------
        a : str
            The new author

        Returns
        -------
        None
        """
        self._author = a

    @property
    def created_at(self):
        """
        Returns the datetime associated with this message.

        Returns
        -------
        datetime.datetime
            Time of creation of post. Could be None if not available/processed.
        """
        return self._created_at

    def set_created_at(self, x):
        """
        Updates the timesttamp for when this message was created.

        Parameters
        ----------
        x : str or float
            The new datetime

        Returns
        -------
        None

        Raises
        ------
        TypeError
            When setting this property with a value that is not a string nor a float.
        """
        if type(x) == str:
            self._created_at = self.parse_datestr(x)
        elif type(x) == float:
            self._created_at = datetime.fromtimestamp(x)
        else:
            raise TypeError(f'Unrecognized created_at conversion: {type(x)} --> {x}')

    @property
    def reply_to(self):
        """
        Returns the unique identifiers of the messages that are replied to by this message.

        Returns
        -------
        set(UID)
            The set of UIDs of the posts this message replies to
        """
        return self._reply_to

    def add_reply_to(self, tid):
        """
        Adds a new UID that this message is replying to.

        Parameters
        ----------
        tid : UID
            The UID to be added

        Returns
        -------
        None
        """
        self._reply_to.add(tid)

    def remove_reply_to(self, tid):
        """
        Removes a UID from the set this message is replying to.

        Parameters
        ----------
        tid : UID
            The UID to be removed
        """
        self._reply_to.remove(tid)

    @property
    def tags(self):
        """
        Returns the tags associated with this message.

        Returns
        -------
        set(str)
            Set of string tags associated with this message
        """
        return self._tags

    def add_tag(self, tag):
        """
        Adds a new tag to this message.

        Parameters
        ----------
        tag : str
            The tag to be added

        Returns
        -------
        None
        """
        self._tags.add(tag)

    def remove_tag(self, tag):
        """
        Removes a tag from this message.

        Parameters
        ----------
        tag : str
            The tag to remove

        Returns
        -------
        None
        """
        self._tags.remove(tag)

    @property
    def platform(self):
        """
        The platform this message was created on

        Returns
        -------
        str
            Platform name
        """
        return self._platform

    @platform.setter
    def platform(self, p):
        """
        Updates the platform this message is from.

        Parameters
        ----------
        p : str
            The platform name

        Returns
        -------
        None
        """
        self._platform = p

    @property
    def lang(self):
        """
        Returns the language this post was written in

        Returns
        -------
        str
            Language code of the message text
        """
        return self._lang

    @lang.setter
    def lang(self, lang):
        """
        Updates the language this post was written in

        Parameters
        ----------
        lang : str
            The language associated with this post

        Returns
        -------
        None
        """
        self._lang = lang

    @staticmethod
    def from_json(data):
        """
        Given an exported JSON object for a Universal Message,
        this function loads the saved data into its fields

        Parameters
        ----------
        data : JSON/dict
            The raw message JSON

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError

    def to_json(self):
        """
        Function for exporting a Universal Post into a JSON object for storage and later use

        Returns
        -------
        JSON/dict
            The JSON formatted UniMessage for disk storage
        """
        return {
            'uid': self._uid,
            'text': self.text,
            'author': self.author,
            'created_at': self.created_at.timestamp() if self.created_at else None,
            'reply_to': list(self.reply_to),
            'platform': self.platform,
            'tags': list(self._tags),
            'lang': self._lang
        }

    def get_mentions(self):
        """
        By default, this will simply return the author
        of the post (if available) for appropriate anonymization

        Returns
        -------
        set(str)
            The mentions detected in this message
        """
        if self.author:
            return {self.author}

        return set()

    def redact(self, redact_map):
        """
        Given a set of terms, this function will properly redact
        all instances of those terms.
        This function is mainly to use for redacting usernames
        or user mentions, so as to protect user privacy.

        Parameters
        ----------
        redact_map : dict(str, str)
            The map of terms and what they should be replaced with

        Returns
        -------
        None
        """
        if self.text:
            for term, replacement in redact_map.items():
                if term in self.text:
                    self.text = re.sub(term, replacement, self.text)

        # Change the author's name if they're in our redaction map
        if self.author in redact_map:
            self.author = redact_map[self.author]

    @property
    def chars(self):
        """
        The number of characters in this message.

        Returns
        -------
        int
            Number of character in the text of this post
        """
        return len(self.text)

    @property
    def tokens(self):
        """
        Returns the text of this message, tokenized.

        Returns
        -------
        list(str)
            List of tokens in this post
        """
        return self._tok.tokenize(self.text)

    @property
    def types(self):
        """
        The unique set of tokens used in this message.

        Returns
        -------
        set(str)
            Set of unique tokens in this post. The vocabulary of this post.
        """
        return set(self.tokens)
