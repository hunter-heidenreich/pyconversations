import re
from datetime import datetime

from .base import UniMessage


class RedditPost(UniMessage):

    """
    Reddit post object with additional Reddit-specific features
    """

    @staticmethod
    def parse_datestr(x):
        return datetime.fromtimestamp(float(x))

    def get_mentions(self):
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
        """
        data['created_at'] = datetime.fromtimestamp(data['created_at']) if data['created_at'] else None
        return RedditPost(**data)

    @staticmethod
    def parse_raw(data, lang_detect=False):
        post_cons = {
            'reply_to': set(),
            'platform': 'Reddit',
            'lang_detect': lang_detect
        }

        ignore_keys = {
            'archived', 'body_html', 'id', 'link_id', 'gilded',
            'ups', 'downs', 'edited', 'controversiality', 'user_reports', 'mod_reports',
            'score', 'subreddit'
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
        cons = {
            'platform': 'Reddit',
            'lang_detect': lang_detect,
            'uid': 't3_' + data['id'],
            'author': data['author'],
            'created_at': RedditPost.parse_datestr(data['created_utc']),
            'tags': {f'board={data["subreddit"]}'}
        }
        if data['type'] == 'comment':
            cons['text'] = data['body']
            cons['reply_to'] = {data['parent_id']}
        elif data['type'] == 'submission':
            cons['text'] = data['title'] + ' ' + data['selftext']
            cons['reply_to'] = set()
        else:
            raise ValueError(f'RedditPost::parse_rd -- Unrecognized type: {data}')

        return RedditPost(**cons)
