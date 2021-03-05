import re
from abc import ABC
from abc import abstractmethod
from datetime import datetime

import gcld3

# Langauge detection module; do not initialize unless asked for!
DETECTOR = None


def get_detector():
    global DETECTOR
    if not DETECTOR:
        DETECTOR = gcld3.NNetLanguageIdentifier(min_num_bytes=0, max_num_bytes=1000)

    return DETECTOR


class UniMessage(ABC):

    """
    The Universal Message class.

    This is designed to be the abstract, baseline object
    that all social media posts / conversation turns
    inherit from.
    The only mandatory field is the uid, a unique field.

    Since this class is abstract, it also has 2 abstract methods that must be implemented
    by all extending classes:
     - parse_raw -- how to go from raw form to structured JSON input to constructor
     - set_created_at -- platform specific time parsing
    """

    def __init__(self, uid, text='', author=None, created_at=None, reply_to=None, platform=None, lang=None, tags=None,
                 lang_detect=False):
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

    def __hash__(self):
        return self._uid

    def __repr__(self):
        return f'UniMessage({self._platform}::{self._author}::{self._created_at}::{self._text[:50]}::tags={",".join(self._tags)})'

    def _detect_language(self):
        if not self._lang and self._lang_detect and self._text:
            res = get_detector().FindLanguage(text=self.text)
            self.lang = res.language if res.is_reliable else 'und'

    @staticmethod
    @abstractmethod
    def parse_raw(raw, lang_detect=False):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def parse_datestr(x):
        raise NotImplementedError

    @property
    def uid(self):
        return self._uid

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, t):
        self._text = t
        self._lang = None
        self._detect_language()

    @property
    def author(self):
        return self._author

    @author.setter
    def author(self, a):
        self._author = a

    @property
    def created_at(self):
        return self._created_at

    def set_created_at(self, x):
        if type(x) == str:
            self._created_at = self.parse_datestr(x)
        elif type(x) == float:
            self._created_at = datetime.fromtimestamp(x)
        else:
            raise TypeError(f'Unrecognized created_at conversion: {type(x)} --> {x}')

    @property
    def reply_to(self):
        return self._reply_to

    def add_reply_to(self, tid):
        self._reply_to.add(tid)

    def remove_reply_to(self, tid):
        self._reply_to.remove(tid)

    @property
    def tags(self):
        return self._tags

    def add_tag(self, tag):
        self._tags.add(tag)

    def remove_tag(self, tag):
        self._tags.remove(tag)

    @property
    def platform(self):
        return self._platform

    @platform.setter
    def platform(self, p):
        self._platform = p

    @property
    def lang(self):
        return self._lang

    @lang.setter
    def lang(self, lang):
        self._lang = lang

    @staticmethod
    def from_json(data):
        """
        Given an exported JSON object for a Universal Message,
        this function loads the saved data into its fields
        """
        raise NotImplementedError

    def to_json(self):
        """
        Function for exporting a Universal Post
        into a JSON object for storage and later use
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
        """
        if self.author:
            return {self.author}

        return set()

    def redact(self, redact_map):
        """
        Given a set of terms,
        this function will properly redact
        all instances of those terms.

        This function is mainly to use for redacting usernames
        or user mentions, so as to protect users
        """
        for term, replacement in redact_map.items():
            self.text = re.sub(term, replacement, self.text)

        # Change the author's name if they're in our redaction map
        if self.author in redact_map:
            self.author = redact_map[self.author]
