==================
Feature Extraction
==================

PyConversations supports two types of feature extraction approaches:

* dictionary extraction - keyed features in a dictionary
* vectorization - direct conversion from posts/conversation(s) to numpy arrays (with optional normalization)

-----------------------
Dictionary Raw Features
-----------------------

Extraction can happen at differently levels
depending on the amount of information
that is available.
For example, there are a set of features that can be extracted
from a post in isolation,
but even more features are available when a post
and the Conversation it exists in are available.

All of these information levels feature a `get_all(...)` function
that gets all available features for that data.
Alternatively, one can use a more specific extraction function like
`get_ints(...)` to get all features that are of integer type.
All of these functions feature optional parameters
`keys=` which specifies a set of feature names that will only be extracted
or `ignore_keys=` which specifies  set of feature names that will be skipped.
All returns are of form `dict(str, Any)` where `Any` is the extracted feature.

^^^^
Post
^^^^

For a post `px`, there exists the following feature functions::

    from pyconversations.feature_extraction.post import *

    get_all(px)  # all features
    get_bools(px)  # all True/False features
    get_categorical(px)  # all features that categorical, e.g. language or platform
    get_counters(px)  # all Counter features (e.g., token frequency)
    get_floats(px)  # all decimal features
    get_ints(px)  # all integer features
    get_substrings(px)  # all list of strings features

^^^^^^^^^^^^^^^^^^^
Post & Conversation
^^^^^^^^^^^^^^^^^^^

For a post `px` and the conversation `cx` it came in, there exists::

    from pyconversations.feature_extraction.post_in_conv import *

    get_all(px, cx)  # all features
    get_bools(px, cx)  # all True/False features
    get_floats(px, cx)  # all floating point features
    get_ints(px, cx)  # all integer features

All of these functions have an additional `include_static=True` parameter
that controls whether the post features (above) are combined with these features.

^^^^^^^^^^^^^^^^^^^
User & Conversation
^^^^^^^^^^^^^^^^^^^

For a user `ux` and the conversation `cx` they participated in, there exists::

    from pyconversations.feature_extraction.user_in_conv import *

    get_all(ux, cx)  # all features
    get_bools(ux, cx)  # all True/False features
    get_floats(ux, cx)  # all floating point features
    get_ints(ux, cx)  # all integer features

For `get_all()` and `get_floats()`,
there are optional `include_post=True` parameters
that compute aggregate statistics (min, max, mean, std. dev., and median)
across the post features of posts in `cx` created by `ux`.


^^^^^^^^^^^^
Conversation
^^^^^^^^^^^^

If you have a conversation `cx`,
there exists the following::

    from pyconversations.feature_extraction.conv import *

    get_all(cx)  # all features
    get_counters(cx)  # all counter features
    get_floats(cx)  # all decimal features
    get_ints(cx)  # all integer features

-------------
Vectorization
-------------

`Vectorizers` are feature extractors that pre-process features into an appropriate format
for machine learning algorithms.
Similar to `sklearn`, this project uses an interface
where `Vectorizers` can be `.fit()` to specific data,
`.transform()` data using the learned paramters (for normalization),
or both (`.fit_transform()`).

All `Vectorizers` take a normalization parameter at construction which can be:

* `None` - No pre-processing; raw, numeric features will be returned
* `minmax` - MinMax scaling will be performed on float and integer features
* `mean` - Mean scaling will be performed on float and integer features
* `standard` - Standard scaling will be performed on float and integer features

^^^^^^^^^^^^^^
PostVectorizer
^^^^^^^^^^^^^^

The PostVectorizer produces a vector for every post in a collection.
Functions can be supplied with post data either through
`posts=<iterable of posts>`,
`conv=<Conversation>,
or `convs=<iterable of Conversations>`.

Additionally, at construction, one can
set `include_conversation=True` or `include_user=True`
to get conversation and/or user information included
within the post vector.

Example::

    from pyconversations.feature_extraction import PostVectorizer

    convos = <iterable of conversations>
    vec = PostVectorizer(normalization='standard', include_conversation=True, include_user=True)
    xs = vec.fit_transform(convs=convos)

If `convos` had `N` posts in total,
`xs` is now a `(N, d)` set of feature vectors
that are scaled to a standard distribution
and include feature information about the conversation
and author for each post.

^^^^^^^^^^^^^^^^^^^^^^
ConversationVectorizer
^^^^^^^^^^^^^^^^^^^^^^

If vectors for conversations are desired instead, then use::

    from pyconversations.feature_extraction import ConversationVectorizer

In addition to a normalization method,
`ConversationVectorizer` also has the following optional construction parameters:

* `agg_post_fts=False` - Include information about post contained within a conversation (in aggregate)
* `agg_user_fts=False` - Include information about users contained within a conversation (in aggregate)
* `include_source_user=True` - Include information about the source user

^^^^^^^^^^^^^^
UserVectorizer
^^^^^^^^^^^^^^

Likewise, there is a vectorizer for user vectors::

    from pyconversations.feature_extraction import UserVectorizer

It features the following (optional) construction parameters:

* `agg_post_fts=False` - Include information about post created by this user (in aggregate)
