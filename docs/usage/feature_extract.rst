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

Each of these levels have an associated static extraction object.
These objects may be used to extract features directly from the data available
in a manner that produces dictionaries where keys are named features
and values are the actual extracted value.
These features are grouped according to their type
and each extraction object returns a (potentially empty) dictionary of features.

Currently supported extract types include:

* bools - True/False values (e.g., is_source)
* categoricals - string categorical variables (e.g., language code)
* counter - Counter objects that represent a distribution (e.g., word frequency distribution for a post/conversation/user)
* floats - floating point features (e.g., aggregate post statistics within a conversation)
* ints - integer features (e.g., post in-degree)
* strs - substrings of interest (e.g., emojis used within a post)

The different levels of extraction are:

* `PostFeatures` - A post in isolation (i.e., no conversational data to enrich post features)
* `PostInConvoFeatures` - A post + conversation
* `ConvoFeatures` - Aggregate conversation features
* `UserInConvoFeatures` - Features about a user within a conversation
* `UserAcrossConvoFeatures` - Features about a user across multiple conversations

All may be imported from `pyconversations.feature_extraction`.
Features are extracted by statically calling the features desired
and supplying the necessary information.
E.g., post boolean features with conversational data::

    PostInConvoFeatures.bools(post, convo)  # returns a dictionary of type dict(str, bool)

-------------
Vectorization
-------------

`Vectorizers` are feature extractors that pre-process features into an appropriate format
for machine learning algorithms.
Similar to `sklearn`, this project uses an interface
where `Vectorizers` can be `.fit()` to specific data,
`.transform()` data using the learned parameters (for normalization),
or both (`.fit_transform()`).

All `Vectorizers` take a normalization parameter at construction which can be:

* `None` - No pre-processing; raw, numeric features will be returned
* `minmax` - MinMax scaling will be performed on float and integer features
* `mean` - Mean scaling will be performed on float and integer features
* `standard` - Standard scaling will be performed on float and integer features

Additionally, vectorizers can return a map from unique ID to row in returned np.array
by adding `include_ids=True` in the `.transform` method.

^^^^^^^^^^^^^^
PostVectorizer
^^^^^^^^^^^^^^

The PostVectorizer produces a vector for every post in a collection.
Functions can be supplied with post data either through
`posts=<iterable of posts>`,
`conv=<Conversation>,
or `convs=<iterable of Conversations>`.

Example::

    from pyconversations.feature_extraction import PostVectorizer

    convos = <iterable of conversations>
    vec = PostVectorizer(normalization='standard')
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

^^^^^^^^^^^^^^
UserVectorizer
^^^^^^^^^^^^^^

Likewise, there is a vectorizer for user vectors::

    from pyconversations.feature_extraction import UserVectorizer

^^^^^^^^^^^^^^^^^
Recommended Usage
^^^^^^^^^^^^^^^^^

A good way to use these vectorizers is to fit the type of vectorizers you want (post, conversation, user)
and then transform each level of data using the `include_ids=True` parameter.
In doing so, you get a vector for each unit (post, conversation, user)
and a map from unique unit identifier to row in the transformed data.

This is very useful for constructing complex input using this package.
For example, one might want to classify every post in a conversation
but may find it beneficial to supply post and user information not only about
the post of interest, but about the parent post as well.
By transforming the data once and retaining this mapping,
you can easily create vectors that correspond to feature sets tuned for the task at hand.

For a concrete example of this,
see the tutorial that applies this idea in a very basic fashion
to the SemEval-2019 task RumourEval:

`Classifying Support Labels (RumourEval from SemEval-2019 Task 7a) <https://github.com/hunter-heidenreich/pyconversations/blob/master/tutorial/RumourEval-SDQC.ipynb>`_
