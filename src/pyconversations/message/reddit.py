import re
from datetime import datetime

from .base import UniMessage


class RedditPost(UniMessage):

    """
    Reddit post object with additional Reddit-specific features
    """

    @staticmethod
    def parse_datestr(x):
        """
        Static method that specifies how to convert the native datetime string
        into a a Python datetime object.

        Parameters
        ----------
        x : str
            The raw datetime string

        Returns
        -------
        datetime.datetime
            The parsed datetime
        """
        return datetime.fromtimestamp(float(x))

    def get_mentions(self):
        """
        Uses Reddit specific regex to attempt to identify
        user mentions within the comment text.

        Returns a set of usernames.

        Returns
        -------
        set(str)
            The set of extracted usernames
        """
        # Reddit mention regex
        names = re.findall(r'/?u/([A-Za-z0-9_-]+)', self.text)
        adj_names = []
        for name in names:
            if '/u/' in name:
                name = name.replace('/u/', '')
            elif 'u/' in name:
                name = name.replace('u/', '')

            adj_names.append(name)

        return super(RedditPost, self).get_mentions() | set(names)

    @staticmethod
    def from_json(data):
        """
        Given an exported JSON object for a Universal Message,
        this function loads the saved data into its fields

        Parameters
        ----------
        data : JSON/dict
            Raw JSON data

        Returns
        -------
        RedditPost
            The loaded post
        """
        data['created_at'] = datetime.fromtimestamp(data['created_at']) if data['created_at'] else None
        return RedditPost(**data)

    @staticmethod
    def parse_raw(data, lang_detect=False):
        """
        Static method that must be implemented by all non-abstract child classes.
        Concrete implementations should specify how to parse the raw data into this object.

        Parameters
        ----------
        data : JSON/dict
            The raw data to be pre-processed.
        lang_detect : bool
            A boolean which specifies whether language detection should be activated. (Default: False)

        Returns
        -------
        RedditPost
            The parsed post
        """
        post_cons = {
            'reply_to': set(),
            'platform': 'Reddit',
            'lang_detect': lang_detect
        }

        ignore_keys = {
            'archived', 'body_html', 'id', 'link_id', 'gilded',
            'ups', 'downs', 'edited', 'controversiality', 'user_reports', 'mod_reports',
            'score', 'subreddit',
            'delta', 'violated_rule'
        }

        for key, value in data.items():
            if key in ignore_keys:
                continue

            if key == 'author_name':
                post_cons['author'] = value
            elif key == 'body':
                post_cons['text'] = post_cons['text'] + ' ' + value if 'text' in post_cons else value
            elif key == 'title':
                post_cons['text'] = value + ' ' + post_cons['text'] if 'text' in post_cons else value
            elif key == 'created':
                post_cons['created_at'] = RedditPost.parse_datestr(value)
            elif key == 'created_utc':
                post_cons['created_at'] = RedditPost.parse_datestr(value)
            elif key == 'name':
                post_cons['uid'] = value
            elif key == 'parent_id':
                post_cons['reply_to'].add(value)
            else:
                raise KeyError(f'RedditPost::parse_raw - Unrecognized key: {key} --> {value}')

        return RedditPost(**post_cons)

    @staticmethod
    def parse_rd(data, lang_detect=True):
        """
        Secondary method for parsing raw Reddit data

        Parameters
        ----------
        data : JSON/dict
            The raw data to be pre-processed.
        lang_detect : bool
            A boolean which specifies whether language detection should be activated. (Default: True)

        Returns
        -------
        RedditPost
            The parsed post
        """
        cons = {
            'platform': 'Reddit',
            'lang_detect': lang_detect,
            'uid': data['id'],  # 't3_' + data['id'],
            'author': data['author'],
            'created_at': RedditPost.parse_datestr(data['created_utc']),
            'tags': {f'board={data["subreddit"]}'}
        }
        if data['type'] == 'comment':
            cons['text'] = data['body']
            pid = data['parent_id']
            for i in range(1, 6):
                pid = pid.replace(f't{i}_', '')
            cons['reply_to'] = {pid}
        elif data['type'] == 'submission':
            cons['text'] = data['title'] + ' ' + data['selftext']
            cons['reply_to'] = set()
        else:
            raise ValueError(f'RedditPost::parse_rd -- Unrecognized type: {data}')

        return RedditPost(**cons)
