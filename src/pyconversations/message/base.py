import re
from abc import ABC
from abc import abstractmethod
from collections import Counter
from datetime import datetime

from ..ld import LangidLangDetect
from ..tokenizers import DefaultTokenizer
from ..tokenizers import LambdaTokenizer
from ..tokenizers import NLTKTokenizer
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


def get_tokenizer(key):
    return {
        'default':     DefaultTokenizer(),
        'NLTK':        NLTKTokenizer(),
        'partitioner': PartitionTokenizer(),
    }[key]


class UniMessage(ABC):
    """
    The Universal Message class.

    This is designed to be the abstract, baseline object
    that all social media posts / conversation turns
    inherit from.
    The only mandatory field is the uid, a unique field.
    """

    MENTION_REGEX = None

    def __init__(self, uid,
                 text='', author=None,
                 created_at=None, reply_to=None, platform=None, lang=None, tags=None,
                 lang_detect=False, tokenizer='partitioner'):
        """
        UniMessage base class initializer

        Parameters
        ----------
        uid : Hashable
            A unique identifier for the post. The only mandatory field.
        text : str
            Text of the message
        author : Hashable
            The author or some identifier thereof
        created_at : datetime.datetime
            The time of creation
        reply_to : set
            A set of UIDs of the posts this message replies to
        platform : str
            The name of the platform, service, etc. where this message was generated
        lang : str
            The language identifier for the language the text of this post is written (or detected to be written) in
        tags : set
            The set of tagged properties
        lang_detect : bool
            Whether or not language detection should be activated when updating post text
        tokenizer : str or lambda(str -> list(str))
            Which tokenizer to use (Default: partitioner)
        """
        # a unique identifier{
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
        self._init_tokenizer()

        self._conv = None
        self._parent = None
        self._children = None

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
    def created_at(self):
        """
        Returns the datetime associated with this message.

        Returns
        -------
        datetime.datetime
            Time of creation of post. Could be None if not available/processed.
        """
        return self._created_at

    @created_at.setter
    def created_at(self, x):
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
    def reply_to(self):
        """
        Returns the unique identifiers of the messages that are replied to by this message.

        Returns
        -------
        set(UID)
            The set of UIDs of the posts this message replies to
        """
        return self._reply_to

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

    def _init_tokenizer(self):
        """
        Sub-selects the tokenizer to use in this class.
        """
        if callable(self._tok):
            self._tok = LambdaTokenizer(self._tok)
        elif type(self._tok) == str:
            # load from dictionary of available choices
            self._tok = get_tokenizer(self._tok)
        else:
            raise ValueError(f'UniMessage._init_tokenizer. Unrecognized value: {self._tok}')

    def _detect_language(self):
        """
        Classifies the text of the post and updates the language field, if asked for.
        """
        if (not self._lang or self.lang == 'und') and self._lang_detect and self._text:
            res = get_detector().get(text=self.text)
            self.lang = res[0] if res[1] >= 0.5 else 'und'

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

    def to_json(self):
        """
        Function for exporting a Universal Post into a JSON object for storage and later use

        Returns
        -------
        JSON/dict
            The JSON formatted UniMessage for disk storage
        """
        return {
            'uid':        self._uid,
            'text':       self.text,
            'author':     self.author,
            'created_at': self.created_at.timestamp() if self.created_at else None,
            'reply_to':   list(self.reply_to),
            'platform':   self.platform,
            'tags':       list(self._tags),
            'lang':       self._lang
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

    def _features_available(self):
        return {
            'uid':         self.uid,
            'author':      self.author,
            'lang':        self.lang,
            'platform':    self.platform,

            'char_len':    self._char_len,

            'tok_len':     self._tok_len,
            'tok_dist':    self._tok_dist,
            'toks':        self._toks,

            'type_len':    self._type_len,
            'types':       self._types,

            'url_cnt':     self._url_cnt,
            'urls':        self._urls,

            'mention_cnt': self._mention_cnt,
            'mentions':    self._mentions,

            'parent_cnt': self._parent_cnt,
        }

    def get_feature(self, key):
        return self._features_available()[key]()

    def _char_len(self):
        """
        The number of characters in this message.

        Returns
        -------
        int
            Number of character in the text of this post
        """
        return len(self._text)

    def _toks(self):
        """
        Tokenizes the text of this message

        Returns
        -------
        list(str)
            The tokenized text
        """
        return self._tok.tokenize(self.text)

    def _tok_len(self):
        """
        The number of tokens in this message.

        Returns
        -------
        int
            Number of tokens in the text of this post
        """
        return len(self._toks())

    def _tok_dist(self):
        """
        The unigram frequency distribution of tokens within this message

        Returns
        -------
        Counter
            A counter of the types and the number of times they occur
        """
        return Counter(self._toks())

    def _types(self):
        """
        The set of unique types in the text of this post.

        Returns
        -------
        set(str)
            The set of unique tokens (types)
        """
        return set(self._toks())

    def _type_len(self):
        """
        The number of unique types within this post

        Returns
        -------
        int
            Number of unique tokens in this post
        """
        return len(self._types())

    def _urls(self):
        """
        Returns the URLs contained within the post

        Returns
        -------
        list(str)
            A list of the URLs identified with regex
        """
        return re.findall(r'(\b(https?|ftp|file)://)[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]', self._text)

    def _url_cnt(self):
        """
        Returns the number of URLs found in this message

        Returns
        -------
        int
            The number of URLs in the text of this message
        """
        return len(self._urls())

    def _mentions(self):
        """
        Returns a list of the mentions within a post

        Returns
        -------
        list(str)
            The list of string user mentions within the post
        """
        if self.MENTION_REGEX is None:
            return []

        return re.findall(self.MENTION_REGEX, self._text)

    def _mention_cnt(self):
        """
        The count of direct mentions within the post

        Returns
        -------
        int
            The count of the number of mentions in this post
        """
        return len(self._mentions())

    def _parent_cnt(self):
        """
        Returns the number of parent posts of this post

        Returns
        -------
        int
            The number of parent posts
        """
        return len(self._reply_to)

    def features(self, features=None):
        """
        Returns features ripe for machine learning or combining through a conversational structure
        (aggregation of higher-order statistics)

        Returns
        -------
        dict(str, ?)
            A dictionary mapping string feature names to their values
        """
        return {
            feat: self.get_feature(feat)
            for feat in self._features_available() if (features is None) or (features is not None and feat in features)
        }
