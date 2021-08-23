===========
Save & Load
===========

This library primarily relies on JSON format,
therefore, both `UniMessage` and `Conversation`
classes can be saved and loaded using `.to_json()` and `.from_json()`, respectively.

A couple quick code "shells" that may be useful for saving/loading data with PyConversations:

---------
JSON File
---------

Assuming you've imported the `json` module:

Saving::

    x = <post | conversation>

    json.dump(x.to_json(), open('...json', 'w+'))

Loading::

    x = Conversation.from_json(open('...json'))
    # or Tweet, RedditPost, etc. instead of Conversation

--------------
JSON Line File
--------------

Assuming you've imported the `json` module.

Saving::

    xs = [<post | conversation>]

    with open('...json', 'w+') as fp:
        for x in xs:
            fp.write(json.dumps(x.to_json()) + '\n')

Loading::

    xs = []
    with open('...json') as fp:
        for line in fp.readlines():
            x = Conversation.from_json(json.loads(line))  # or Tweet, etc.
            xs.append(x)
