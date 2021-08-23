================
UniMessage Class
================

A `UniMessage` is the underlying message/post/social media object within PyConversations.
As of v.0.0.3, there are four concrete implementations of `UniMessage`:

* Tweet (Twitter)
* RedditPost (Reddit)
* ChanPost (4chan)
* FBPost (Facebook)

These capture properties like time representation, raw format reading, specific platform tagging,
and user regex (if applicable, Twitter & Reddit).
For the remainder of this introduction,
we shall use the concrete `Tweet` class.
However, note that the operations discussed/detailed here are applicable to all classes
that inherit `UniMessage`.

-------------
Instantiation
-------------

There are three ways to create a `UniMessage`.
First, there is raw instantiation by filling out constructor parameters::

    t = Tweet(UID, text='...', author='...',
              created_at=datetime.datetime(...),
              reply_to=[...UIDs of posts replied to...], lang='...')

Only the `UID` field is required; all others are set to null values by default.
Here, some of the most common parameters are enumerated but there are a few more that may be useful,
so check out the full documentation if interested.

Second, there is re-construction from a re-loaded JSON representation::

    t = Tweet.from_json(RAW_JSON)

Finally, for some `UniMessage` implementations there are functions that read raw JSON data
as output directly from the platform::

    t = Tweet.parse_raw(RAW, detect_language=True)

Though, if using these, see their specific documentation pages
for more information.

----------
Properties
----------

A `UniMessage` has several key properties:

* `UID` - a unique identifier for the post
* `text` - the string message associated with the post
* `author` - the name/ID of the author of the message
* `created_at` - the time the post was created
* `reply_to` - a set of UIDs of the posts that this object references/replies to
* `tags` - a set of tags attached to this post
* `platform` - the string name of the platform associated with this post
* `lang` - the language code of the text of the post as given or detected
* `tokens` - a tokenized list using the selected tokenizer and the message of this post

For `reply_to` and `tags`, we have `add_*` and `remove_*` methods for managing their members::

    post.add_tag('test_tag')
    post.remove_tag('test_tag')

-------------------------
User Mentions & Redaction
-------------------------

A set of the authors mentioned (if supported) can be generated with::

    post.get_mentions()

This is useful for privacy and redaction purposes when used with the
`.redact()` function which takes a set of terms and what they should be mapped to::

    term_map = {m: 'USER' for m in post.get_mentions()}  # map from user names to redaction token
    post.redact(term_map)

------------------------
Alternate Representation
------------------------

`UniMessage` can be exported to a JSON format that can be
reloaded later::

    post = post.from_json(post.to_json())  # invariance over this operation

--------------------------
Parsing Platform Datetimes
--------------------------

Where applicable, concrete implementations have a
static function for parsing platform specific timestamps
into a Python datetime representation::

    dt = Tweet.parse_datestr(RAW)
