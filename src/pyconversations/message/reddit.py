import re
from datetime import datetime

from .base import UniMessage


class RedditPost(UniMessage):
    """
    Reddit post object with additional Reddit-specific features
    """

    MENTION_REGEX = r'(^|[^\w])/?u/([A-Za-z0-9_-]+)\b'
    CLASS_STR = 'RedditPost'

    def __init__(self, **kwargs):

        kwargs['platform'] = 'Reddit'

        super(RedditPost, self).__init__(**kwargs)

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
        names = re.findall(self.MENTION_REGEX, self.text)
        names = [n[1] for n in names]

        return super(RedditPost, self).get_mentions() | set(names)

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
        list(RedditPost)
            The parsed post
        """
        if 'kind' in data and 'data' in data:  # and data['kind'] == 'Listing':
            if data['kind'] == 'Listing':
                return [x for child in data['data']['children'] for x in
                        RedditPost.parse_raw(child['data'], lang_detect=lang_detect)]
            else:
                return [x for x in RedditPost.parse_raw(data['data'], lang_detect=lang_detect)]

        if type(data) == list:
            return [RedditPost.parse_raw(x, lang_detect=lang_detect) for x in data if type(x) == dict]

        post_cons = {
            'reply_to':    set(),
            'lang_detect': lang_detect
        }

        ignore_keys = {
            'approved_at_utc',
            'approved_by',
            'archived',
            # 'author',
            'author_cakeday',
            'author_flair_background_color',
            'author_flair_css_class',
            'author_flair_richtext',
            'author_flair_template_id',
            'author_flair_text',
            'author_flair_text_color',
            'author_flair_type',
            # 'author_name',
            'banned_at_utc',
            'banned_by',
            # 'body',
            'body_html',
            'can_gild',
            'can_mod_post',
            'category',
            # 'children',
            'clicked',
            'collapsed',
            'collapsed_reason',
            'content_categories',
            'contest_mode',
            'controversiality',
            'count',
            # 'created',
            # 'created_utc',
            'delta',
            'depth',
            'distinguished',
            'domain',
            'downs',
            'edited',
            'gilded',
            'hidden',
            'hide_score',
            # 'id',
            'is_crosspostable',
            'is_meta',
            'is_original_content',
            'is_reddit_media_domain',
            'is_self',
            'is_submitter',
            'is_video',
            'kind',
            'likes',
            'link_flair_background_color',
            'link_flair_css_class',
            'link_flair_richtext',
            'link_flair_template_id',
            'link_flair_text',
            'link_flair_text_color',
            'link_flair_type',
            'link_id',
            'locked',
            'media',
            'media_embed',
            'media_only',
            'mod_note',
            'mod_reason_by',
            'mod_reason_title',
            'mod_reports',
            # 'name',
            'no_follow',
            'num_comments',
            'num_crossposts',
            'num_reports',
            'over_18',
            # 'parent_id',
            'parent_whitelist_status',
            'permalink',
            'pinned',
            'post_categories',
            'post_hint',
            'preview',
            'previous_visits',
            'pwls',
            'quarantine',
            'removal_reason',
            # 'replies',
            'report_reasons',
            'rte_mode',
            'saved',
            'score',
            'score_hidden',
            'secure_media',
            'secure_media_embed',
            # 'selftext',
            'selftext_html',
            'send_replies',
            'spoiler',
            'stickied',
            'subreddit',
            'subreddit_id',
            'subreddit_name_prefixed',
            'subreddit_subscribers',
            'subreddit_type',
            'suggested_sort',
            'thumbnail',
            'thumbnail_height',
            'thumbnail_width',
            # 'title',
            'ups',
            'upvote_ratio',
            'url',
            'user_reports',
            'view_count',
            'violated_rule',
            'visited',
            'whitelist_status',
            'wls'
        }

        out = []
        for key, value in data.items():
            if key in ignore_keys:
                continue

            if key == 'author_name' or key == 'author':
                post_cons['author'] = value
            elif key == 'body' or key == 'selftext':
                post_cons['text'] = post_cons['text'] + '\n' + value if 'text' in post_cons else value
            elif key == 'title':
                post_cons['text'] = value + '\n' + post_cons['text'] if 'text' in post_cons else value
            elif key == 'created':
                post_cons['created_at'] = RedditPost.parse_datestr(value)
            elif key == 'created_utc':
                if 'created' in data:
                    continue
                post_cons['created_at'] = RedditPost.parse_datestr(value)
            elif key == 'id':
                post_cons['uid'] = value
            elif key == 'name':
                if 'id' not in data:
                    post_cons['uid'] = value
            elif key == 'parent_id':
                if re.search(r't\d_', value):
                    value = re.sub(r't\d_', '', value)
                post_cons['reply_to'].add(value)
            elif key == 'replies' or key == 'children':
                if value:
                    out.extend(RedditPost.parse_raw(value, lang_detect=lang_detect))
            else:
                print(data.keys())
                raise KeyError(f'RedditPost::parse_raw - Unrecognized key: {key} --> {value}')

        if post_cons['uid'] != '_':
            out.append(RedditPost(**post_cons))
        return out

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
            'lang_detect': lang_detect,
            'uid':         data['id'],  # 't3_' + data['id'],
            'author':      data['author'],
            'created_at':  RedditPost.parse_datestr(data['created_utc']),
            'tags':        {f'board={data["subreddit"]}'}
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
