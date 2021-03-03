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
        data['created_at'] = datetime.fromtimestamp(data['created_at'])
        return FBPost(**data)

    def set_created_at(self, x):
        if type(x) == str:
            self._created_at = FBPost.parse_datestr(x)
        elif type(x) == float:
            self._created_at = datetime.fromtimestamp(x)
        else:
            raise TypeError(f'Unrecognized created_at conversion: {type(x)} --> {x}')

    @staticmethod
    def parse_raw(data, post_type='post', in_reply_to=None):
        if post_type == 'post':
            return FBPost.parse_raw_post(data)
        elif post_type == 'comments':
            return FBPost.parse_raw_comments(data, in_reply_to=in_reply_to)
        elif post_type == 'replies':
            return FBPost.parse_raw_replies(data, in_reply_to=in_reply_to)
        else:
            raise ValueError(f'FBPost::parse_raw - Unrecognized post_type: {post_type}')

    @staticmethod
    def parse_raw_post(data):
        post_cons = {
            'platform': 'Facebook',
        }

        ignore_keys = {
            'caption', 'link', 'picture',
            'shares', 'updated_time', 'replies', 'story',
            'source', 'type', 'first_party', 'place'
        }
        for key, value in data.items():
            if key in ignore_keys:
                continue

            if key == 'created_time':
                post_cons['created_at'] = FBPost.parse_datestr(value)
            elif key == 'description':
                post_cons['text'] = (post_cons['text'] if 'text' in post_cons else '') + value
            elif key == 'message':
                post_cons['text'] = (post_cons['text'] if 'text' in post_cons else '') + value
            elif key == 'name':
                post_cons['author'] = value
            elif key == 'id':
                post_cons['uid'] = value
            else:
                raise KeyError(f'FBPost::parse_raw_post - Unrecognized key in FB raw post: {key} --> {value}')

        return FBPost(**post_cons)

    @staticmethod
    def parse_raw_comments(data, in_reply_to=None):
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
                'platform': 'Facebook'
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
    def parse_raw_replies(data, in_reply_to=None):
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
                'platform': 'Facebook'
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
                out.extend(FBPost.parse_raw_replies(comment['replies'], in_reply_to=post_cons['uid']))

            out.append(FBPost(**post_cons))
        return out
