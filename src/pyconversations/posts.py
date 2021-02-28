import re

from abc import ABC, abstractmethod
from datetime import datetime


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

    def __init__(self, uid, text='', author=None, created_at=None, reply_to=None, platform=None, lang=None, tags=None):
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

    def __hash__(self):
        return self._uid

    def __repr__(self):
        return f'UniMessage({self._platform}::{self._author}::{self._created_at}::{self._text[:50]}, tags={self._tags})'

    @staticmethod
    @abstractmethod
    def parse_raw(raw):
        pass

    @property
    def uid(self):
        return self._uid

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, t):
        self._text = t

    @property
    def author(self):
        return self._author

    @author.setter
    def author(self, a):
        self._author = a

    @property
    def created_at(self):
        return self._created_at

    @abstractmethod
    def set_created_at(self, x):
        pass

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


class Tweet(UniMessage):

    """
    Twitter post object with additional Twitter-specific features
    """

    @staticmethod
    def parse_datestr(x):
        return datetime.strptime(x, '%a %b %d %H:%M:%S +0000 %Y')

    def get_mentions(self):
        # twitter mention regex
        names = re.findall(r'@([^\s:]+)', self.text)

        return super(Tweet, self).get_mentions() | set(names)

    @staticmethod
    def from_json(data):
        """
        Given an exported JSON object for a Universal Message,
        this function loads the saved data into its fields
        """
        data['created_at'] = datetime.fromtimestamp(data['created_at'])
        return Tweet(**data)

    def set_created_at(self, x):
        if type(x) == str:
            self._created_at = Tweet.parse_datestr(x)
        elif type(x) == float:
            self._created_at = datetime.fromtimestamp(x)
        else:
            raise TypeError(f'Unrecognized created_at conversion: {type(x)} --> {x}')

    @staticmethod
    def parse_raw(data):
        """
        Takes a raw tweet and returns
        :param data:
        :return:
        """

        cons_vals = {
            'platform': 'Twitter',
            'reply_to': set()
        }
        out = []

        ignore_keys = {
            'id_str', 'truncated', 'display_text_range', 'entities', 'source',
            'in_reply_to_status_id_str', 'in_reply_to_user_id', 'in_reply_to_user_id_str',
            'in_reply_to_screen_name', 'geo', 'coordinates', 'place', 'contributors',
            'is_quote_status', 'retweet_count', 'favorite_count', 'favorited',
            'retweeted', 'metadata', 'extended_entities', 'possibly_sensitive',
            'quoted_status_id_str', 'quoted_status_permalink', 'withheld_in_countries',
            'in_reply_to_status_created_at', 'possibly_sensitive_appealable', 'scopes',
            'withheld_scope', 'withheld_copyright'
        }
        for key, value in data.items():
            if key in ignore_keys:
                continue

            if key == 'created_at':
                cons_vals['created_at'] = Tweet.parse_datestr(value)
            elif key == 'id':
                cons_vals['uid'] = value
            elif key == 'full_text' and 'text' not in cons_vals:
                cons_vals['text'] = value
            elif key == 'text' and 'text' not in cons_vals:
                cons_vals['text'] = value
            elif key == 'lang':
                cons_vals['lang'] = value
            elif key == 'in_reply_to_status_id':
                cons_vals['reply_to'].add(value)
            elif key == 'quoted_status_id':
                cons_vals['reply_to'].add(value)
            elif key == 'user':
                cons_vals['author'] = value['screen_name']
            elif key == 'quoted_status':
                out.extend(Tweet.parse_raw(value))
            else:
                raise KeyError(f'Tweet:parse_raw - Unrecognized key: {key} --> {value}')

        # Do entities last
        if 'entities' in data:
            ignore_keys = {
                'hashtags', 'symbols', 'user_mentions'
            }
            for key, value in data['entities'].items():
                if key in ignore_keys:
                    continue

                if key == 'media':
                    for v in value:
                        cons_vals['text'] = re.sub(v['url'], v['display_url'], cons_vals['text'])
                elif key == 'urls':
                    for v in value:
                        cons_vals['text'] = re.sub(v['url'], v['expanded_url'], cons_vals['text'])
                else:
                    raise KeyError(f'Tweet:parse_raw - Unrecognized key: {key} --> {value}')

        if 'text' in cons_vals:
            out.append(Tweet(**cons_vals))

        return out
