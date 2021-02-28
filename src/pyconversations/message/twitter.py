import re

from datetime import datetime

from .base import UniMessage


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
