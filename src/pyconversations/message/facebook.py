from datetime import datetime

from .base import UniMessage


class FBPost(UniMessage):

    """
    FB-specific FB Post object with Facebook specific features
    """

    CLASS_STR = 'FBPost'

    def __init__(self, **kwargs):

        kwargs['platform'] = 'Facebook'

        super(FBPost, self).__init__(**kwargs)

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
        return datetime.strptime(x, '%Y-%m-%dT%H:%M:%S+0000')

    @staticmethod
    def parse_raw(data, post_type='post', in_reply_to=None, lang_detect=False):
        """
        Static method that must be implemented by all non-abstract child classes.
        Concrete implementations should specify how to parse the raw data into this object.

        Parameters
        ----------
        data : JSON/dict
            The raw data to be pre-processed.
        post_type : str
            Facebook specific feature; what type of FB object to parse: ['post', 'comments', 'replies']
        in_reply_to : set(UID)
            Unique IDs of posts this object is replying to
        lang_detect : bool
            A boolean which specifies whether language detection should be activated. (Default: False)

        Returns
        -------
        FBPost or list(FBPost)
            The parsed posts

        Raises
        ------
        ValueError
            When given an invalid `post_type`
        """
        if post_type == 'post':
            return FBPost.parse_raw_post(data, in_reply_to=None, lang_detect=lang_detect)
        elif post_type == 'comments':
            return FBPost.parse_raw_comments(data, in_reply_to=in_reply_to, lang_detect=lang_detect)
        elif post_type == 'replies':
            return FBPost.parse_raw_replies(data, in_reply_to=in_reply_to, lang_detect=lang_detect)
        else:
            raise ValueError(f'FBPost::parse_raw - Unrecognized post_type: {post_type}')

    @staticmethod
    def parse_raw_post(data, lang_detect=False, in_reply_to=None):
        """
        Parses the raw JSON of a FB post object

        Parameters
        ----------
        data : JSON/dict
            The raw data to be pre-processed.
        in_reply_to : set(UID)
            Unique IDs of posts this object is replying to
        lang_detect : bool
            A boolean which specifies whether language detection should be activated. (Default: False)

        Returns
        -------
        FBPost
            The parsed post
        """
        if not data:
            return

        post_cons = {
            'lang_detect': lang_detect,
            'author': in_reply_to,
            'text': '',
            'tags': {'type=' + data['type']} if 'type' in data else set()
        }

        for key in ['name', 'description', 'story', 'caption', 'message']:
            if key in data:
                post_cons['text'] += ' ' + data[key]
        post_cons['text'] = post_cons['text'].strip()

        ignore_keys = {
            'caption', 'link', 'picture',
            'shares', 'updated_time', 'replies', 'story',
            'source', 'type', 'first_party', 'place',
            'name', 'description', 'message'
        }
        for key, value in data.items():
            if key in ignore_keys:
                continue

            if key == 'created_time':
                post_cons['created_at'] = FBPost.parse_datestr(value)
            elif key == 'id':
                post_cons['uid'] = value
            else:
                raise KeyError(f'FBPost::parse_raw_post - Unrecognized key in FB raw post: {key} --> {value}')

        return FBPost(**post_cons)

    @staticmethod
    def parse_raw_comments(data, in_reply_to=None, lang_detect=False):
        """
        Parses the raw JSON of a FB comments object.
        Returns a list of parsed comments.

        Parameters
        ----------
        data : JSON/dict
            The raw data to be pre-processed.
        in_reply_to : set(UID)
            Unique IDs of posts this object is replying to
        lang_detect : bool
            A boolean which specifies whether language detection should be activated. (Default: False)

        Returns
        -------
        list(FBPost)
            The parsed comments
        """
        out = []

        if not data:
            return out

        ignore_keys = {
            'response', 'from'
        }

        if type(data) == dict:
            data = data['data']

        for comment in data:
            post_cons = {
                'lang_detect': lang_detect,
            }

            if in_reply_to:
                post_cons['reply_to'] = {in_reply_to}

            for key, value in comment.items():
                if key in ignore_keys:
                    continue

                if key == 'id':
                    post_cons['uid'] = value
                elif key == 'message':
                    post_cons['text'] = value
                elif key == 'created_time':
                    post_cons['created_at'] = FBPost.parse_datestr(value)
                elif key == 'userID':
                    post_cons['author'] = value
                else:
                    raise KeyError(f'FBPost::parse_raw_comments - Unrecognized key in FB raw comment: {key} --> {value}')

            out.append(FBPost(**post_cons))

        return out

    @staticmethod
    def parse_raw_replies(data, in_reply_to=None, lang_detect=False):
        """
        Parses the raw JSON of a FB replies object.
        Returns a list of parsed comments.

        Parameters
        ----------
        data : JSON/dict
            The raw data to be pre-processed.
        in_reply_to : set(UID)
            Unique IDs of posts this object is replying to
        lang_detect : bool
            A boolean which specifies whether language detection should be activated. (Default: False)

        Returns
        -------
        list(FBPost)
            The parsed replies
        """
        out = []

        if not data:
            return out

        ignore_keys = {
            'response', 'from'
        }

        if type(data) == dict:
            data = data['data']

        for comment in data:
            post_cons = {
                'lang_detect': lang_detect,
            }

            if in_reply_to:
                post_cons['reply_to'] = {in_reply_to}

            for key, value in comment.items():
                if key in ignore_keys:
                    continue

                if key == 'id':
                    post_cons['uid'] = value
                elif key == 'message':
                    post_cons['text'] = value
                elif key == 'created_time':
                    post_cons['created_at'] = FBPost.parse_datestr(value)
                elif key == 'userID':
                    post_cons['author'] = value
                elif key == 'replies':
                    continue
                else:
                    raise KeyError(f'FBPages::parse_raw_replies - Unrecognized key in FB raw reply: {key} --> {value}')

            if 'replies' in comment and comment['replies']:
                rs = FBPost.parse_raw_replies(comment['replies'], in_reply_to=post_cons['uid'], lang_detect=lang_detect)
                out.extend(rs)

            out.append(FBPost(**post_cons))
        return out
