import html
import re
from datetime import datetime

from .base import UniMessage


class ChanPost(UniMessage):

    """
    4chan post object with additional 4chan-specific features
    """

    CLASS_STR = '4chanPost'
    MENTION_REGEX = r'>>(\d+)'

    def __init__(self, **kwargs):

        kwargs['platform'] = '4chan'

        super(ChanPost, self).__init__(**kwargs)

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
            Creation datetime of 4chan timestamp
        """
        return datetime.fromtimestamp(float(x))

    @staticmethod
    def exclude_replies(comment):
        """
        Function to remove quotes from a reply and return reference to the posts that were replied to

        Parameters
        ----------
        comment : str
            Text from 4chan comment

        Returns
        -------
        2-tuple(str, list(int))
            The comment and the list of identified references
        """
        refs = re.findall(r'>>(\d+)', comment)
        return comment, refs

    @staticmethod
    def clean_text(comment):
        """
        Cleans the raw HTML of a cached 4chan post,
        returning both the references and teh comment itself

        Parameters
        ----------
        comment : str
            The raw text of a 4chan comment

        Returns
        -------
        2-tuple(str, list(int))
            The comment text and a list of identified reference IDs
        """
        comment = html.unescape(comment)
        comment = re.sub(r"<br/?>", "\n", comment)
        comment = re.sub(r"<wbr/?>", "", comment)
        comment, rfs = ChanPost.exclude_replies(comment)
        comment = str(comment).strip()

        return comment, rfs

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
        ChanPost
            The parsed post
        """
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
            'reply_to':   reps,
            'lang_detect': lang_detect
        })
