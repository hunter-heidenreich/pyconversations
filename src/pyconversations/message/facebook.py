from datetime import datetime

from .base import UniMessage


class FBPost(UniMessage):

    @staticmethod
    def parse_datestr(x):
        return datetime.strptime(x, '%Y-%m-%dT%H:%M:%S+0000')

    @staticmethod
    def from_json(data):
        """
        Given an exported JSON object for a Universal Message,
        this function loads the saved data into its fields
        """
        data['created_at'] = datetime.fromtimestamp(data['created_at']) if data['created_at'] else None
        return FBPost(**data)

    @staticmethod
    def parse_raw(data, post_type='post', in_reply_to=None, lang_detect=False):
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
        if not data:
            return

        post_cons = {
            'platform': 'Facebook',
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
                'platform': 'Facebook',
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
                'platform': 'Facebook',
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
