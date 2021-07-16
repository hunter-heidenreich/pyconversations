
Changelog
=========

0.0.3
------------------

* Reading raw formats
    * Update known keys in reading raw tweet data
    * Update (and greatly expand) the known keys for extracting data from raw Reddit data
* Filtration
    * Conversation.filter now returns a new Conversation object without modifying the original object
    * Add ability to filter by platform
* Message objects
    * Set platforms in specific message type constructors
    * Upgrade user detection regex (Twitter and Reddit)
* Tokenizer
    * Alter tokenizers into an object centered format (no longer a functional one)
    * Add a wrapper for generic tokenizer types

0.0.2 (2021-07-01)
------------------

* Fix pyconversations.tokenizers.PartitionTokenizer

0.0.1 (2021-06-30)
------------------

* Stable first release on PyPI, with properly formatted and distributable dependencies.
* Jupyter Notebook tutorials

0.0.0 (2021-05-30)
------------------

* First release on PyPI.
