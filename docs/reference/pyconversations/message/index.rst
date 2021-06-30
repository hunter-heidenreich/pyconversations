pyconversations.message
=======================

`pyconversations.message` contains the definitions for messages on a variety of platforms.
Currently supported are Twitter, Reddit, 4chan, and Facebook.
All classes inherit from `UniMessage` as a universal template/data model.
This model can (and will!) be extended to future platforms.

.. toctree::
    :glob:

    UniMessage*
    chan*
    fb*
    reddit*
    twitter*
