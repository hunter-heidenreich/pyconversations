import re
from datetime import datetime

from .base import UniMessage


class Tweet(UniMessage):

    """
    Twitter post object with additional Twitter-specific features
    """

    @staticmethod
    def parse_datestr(x):
        """
        Static method that specifies how to convert the native datetime string
        into a a Python datetime object.

        Parameters
        ----------
        x
            The raw datetime string
        """
        return datetime.strptime(x, '%a %b %d %H:%M:%S +0000 %Y')

    def get_mentions(self):
        """
        Uses Twitter specific regex to attempt to identify
        user mentions within the comment text.

        Returns a set of usernames.
        """
        # twitter mention regex
        names = re.findall(r'@[a-zA-Z0-9_]{1,15}', self.text)
        names = [name[1:] for name in names]

        return super(Tweet, self).get_mentions() | set(names)

    @staticmethod
    def from_json(data):
        """
        Given an exported JSON object for a Universal Message,
        this function loads the saved data into its fields

        Parameters
        ----------
        data
            Raw JSON data
        """
        data['created_at'] = datetime.fromtimestamp(data['created_at']) if data['created_at'] else None
        return Tweet(**data)

    @staticmethod
    def parse_raw(data, lang_detect=False):
        """
        Static method that must be implemented by all non-abstract child classes.
        Concrete implementations should specify how to parse the raw data into this object.
        Returns a list of Twitter posts.

        Parameters
        ----------
        data
            The raw data to be pre-processed.
        lang_detect
            A boolean which specifies whether language detection should be activated. (Default: False)
        """
        cons_vals = {
            'platform': 'Twitter',
            'reply_to': set(),
            'lang_detect': lang_detect
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
            elif key == 'full_text':
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
                        cons_vals['text'] = re.sub(v['url'], v['display_url'], cons_vals.get('text', ''))
                elif key == 'urls':
                    for v in value:
                        cons_vals['text'] = re.sub(v['url'], v['expanded_url'], cons_vals.get('text', ''))
                else:
                    raise KeyError(f'Tweet:parse_raw - Unrecognized key: {key} --> {value}')

        if 'text' in cons_vals and cons_vals['text']:
            out.append(Tweet(**cons_vals))

        return out
