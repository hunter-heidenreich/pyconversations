==================
Conversation Class
==================

The `Conversation` class is the basic storage container
for all `UniMessage` objects.
It can be imported with::

    from pyconversations.convo import Conversation

-------------
Instantiation
-------------

A `Conversation` is created using its constructor, which can take two optional
parameters: `posts` (a dictionary of `UniMessage` objects) and `convo_id` (a string identifier)::

    convo = Conversation(convo_id='tutorial_convo')

Alternatively, if you have a JSON representation of the Conversation (something the class exports to),
you can statically instantiate a Conversation using::

    convo = Conversation.from_json(<JSON_DATA_HERE>)

-----------------------
Conversation Arithmetic
-----------------------

For ease of combining collections of posts, the `Conversation` overloads the addition operator::

    all_posts = convo_a + convo_b  # all posts (e.g., the union of) in the two conversations

-------------------------
Adding and removing posts
-------------------------

Posts are added to a conversation by supplying a `UniMessage` object::

    convo.add_post(post_object)

Likewise, by supplying the UID of a post within the conversation,
a post can be removed::

    convo.remove_post(uid)

--------------------------
User-Information Redaction
--------------------------

For Twitter and Reddit, user-regex strings are available
that allow for the redaction of usernames and user mentions (within message text).
This can be done in a way that retains all unique users, assigning them an integer ID::

    convo.redact()

Here, the first author identified is mapped to `USER0` and the Nth user is mapped to `USER(N-1)`.
This allows for an anonymized retention of unique user information to track which users said what and to whom.
It can be performed at a per-Conversation level or can be merged into a mega Conversation to perform redaction
at a corpus-wide level.

Alternatively, if unique user information is unnecessary or undesired, one can map all author names and
user mentions to a `USER` token with::

    convo.redact(assign_ints=False)

-------------------------
Alternate Representations
-------------------------

To convert a Conversation into a JSON representation that can be re-loaded later, try::

    convo.to_json()

If desired, one can also get a networkx.Graph representation of the Conversation with::

    convo.as_graph()

----------------------
Splicing Conversations
----------------------

A Conversation's posts can be filtered using::

    sub = convo.filter(**args)

which produces a new Conversation object from the posts
within `convo` that meet the criteria specified in `args`.

Specifically, `.filter()` makes the following keys available:

* `by_langs` -- a set of language codes desired
* `min_chars` -- a minimum length of text
* `before` -- a max datetime criteria
* `after` -- a min datetime criteria
* `by_tags` -- a set of post tags
* `by_platform` -- a set of platform names
* `by_author` -- a single author name

Additionally, some common splits are made available through::

    convo.segment()  # Splits into a list of disjoint (non-cross-referencing) sub-conversations

    convo.get_parents(uid)  # post(s) that UID is replying to

    convo.get_children(uid)  # post(s) that reply to UID

    convo.get_before(uid)  # all posts that occurred in the conversation before UID

    convo.get_after(uid)  # all posts that occurred in the conversation after UID

    convo.get_ancestors(uid)  # all posts on the path from UID's parent(s) to the root post

    convo.get_descendants(uid)  # all posts in UID's child sub-tree(s)

When these do not suffice, `.filter()` is your go-to for custom filter operations.

----------------------------
Other Methods and Properties
----------------------------

Conversations have several other methods that may prove useful to access.

* `.get_sources()` - Returns the set of source posts (posts that do not reply to any other)
* `.time_order()` - Returns a list of the UIDs of posts in time-sorted order
* `.text_stream()` - Returns a list of the text fields of the posts in time-sorted order (e.g., collapses DAG structure into a linear ordering)

Finally, Conversations have two underlying properties:

* `.convo_id` - A string identifier for the conversation (potentially auto-generated from source IDs)
* `.posts` - The underlying dictionary of stored post objects
