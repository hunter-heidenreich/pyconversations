import html
import re

from datetime import datetime

from .base import UniMessage


class ChanPost(UniMessage):

    """
    4chan post object with additional 4chan-specific features
    """

    @staticmethod
    def parse_datestr(x):
        return datetime.fromtimestamp(float(x))

    @staticmethod
    def from_json(data):
        """
        Given an exported JSON object for a Universal Message,
        this function loads the saved data into its fields
        """
        data['created_at'] = datetime.fromtimestamp(data['created_at'])
        return ChanPost(**data)

    @staticmethod
    def exclude_replies(comment):
        """
        Function to remove quotes from a reply
        and return reference to the posts that
        were replied to
        """
        refs = re.findall('>>(\d+)', comment)

        lines = comment.split("\n")
        lines = filter(lambda x: not bool(re.match(">>(\d+)", x.strip())), lines)
        comment = "\n".join(lines)
        comment = re.sub(">>(\d+)", "", comment)

        return comment, refs

    @staticmethod
    def clean_text(comment):
        """
        Cleans the raw HTML of a cached 4chan post,
        returning both the references and teh comment itself
        """
        comment = html.unescape(comment)
        comment = re.sub("<w?br/?>", "\n", comment)
        comment = re.sub("<a href=\".+\" class=\"(\w+)\">", " ", comment)
        comment = re.sub("</a>", " ", comment)
        comment = re.sub("<span class=\"(\w+)\">", " ", comment)
        comment = re.sub("</span>", " ", comment)
        comment = re.sub("<pre class=\"(\w+)\">", " ", comment)
        comment = re.sub("</pre>", " ", comment)

        comment, rfs = ChanPost.exclude_replies(comment)

        comment = re.sub("[^\x00-\x7F]", " ", comment)

        comment = re.sub("&(amp|lt|gt|ge|le)(;|)", " ", comment)

        comment = re.sub("\\s\\s+", " ", comment)
        comment = re.sub("\n", " ", comment)
        comment = str(comment).strip()

        return comment, rfs

    @staticmethod
    def parse_raw(data, lang_detect=False):
        if 'com' not in data:
            return

        txt, rfs = ChanPost.clean_text(data['com'])

        reps = {int(data['resto'])} if data['resto'] else set()
        reps |= set([int(x) for x in rfs])

        if int(data['no']) in reps:
            reps.remove(int(data['no']))

        return ChanPost(**{
            'uid':        int(data['no']),
            'created_at': datetime.fromtimestamp(data['time']),
            'text':       txt,
            'author':     data['name'] if 'name' in data else None,
            'platform':   '4Chan',
            'reply_to':   reps,
            'lang_detect': lang_detect
        })
