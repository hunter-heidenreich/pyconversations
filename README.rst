========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |travis| |appveyor| |requires|
        | |codecov|
    * - package
      - | |version| |wheel| |supported-versions| |supported-implementations|
        | |commits-since|
.. |docs| image:: https://readthedocs.org/projects/pyconversations/badge/?style=flat
    :target: https://readthedocs.org/projects/pyconversations
    :alt: Documentation Status

.. |travis| image:: https://api.travis-ci.com/hunter-heidenreich/pyconversations.svg?branch=master
    :alt: Travis-CI Build Status
    :target: https://travis-ci.com/github/hunter-heidenreich/pyconversations

.. |appveyor| image:: https://ci.appveyor.com/api/projects/status/github/hunter-heidenreich/pyconversations?branch=master&svg=true
    :alt: AppVeyor Build Status
    :target: https://ci.appveyor.com/project/hunter-heidenreich/pyconversations

.. |requires| image:: https://requires.io/github/hunter-heidenreich/pyconversations/requirements.svg?branch=master
    :alt: Requirements Status
    :target: https://requires.io/github/hunter-heidenreich/pyconversations/requirements/?branch=master

.. |codecov| image:: https://codecov.io/gh/hunter-heidenreich/pyconversations/branch/master/graphs/badge.svg?branch=master
    :alt: Coverage Status
    :target: https://codecov.io/github/hunter-heidenreich/pyconversations

.. |version| image:: https://img.shields.io/pypi/v/pyconversations.svg
    :alt: PyPI Package latest release
    :target: https://pypi.org/project/pyconversations

.. |wheel| image:: https://img.shields.io/pypi/wheel/pyconversations.svg
    :alt: PyPI Wheel
    :target: https://pypi.org/project/pyconversations

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/pyconversations.svg
    :alt: Supported versions
    :target: https://pypi.org/project/pyconversations

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/pyconversations.svg
    :alt: Supported implementations
    :target: https://pypi.org/project/pyconversations

.. |commits-since| image:: https://img.shields.io/github/commits-since/hunter-heidenreich/pyconversations/v0.0.0.svg
    :alt: Commits since latest release
    :target: https://github.com/hunter-heidenreich/pyconversations/compare/v0.0.0...master



.. end-badges

A package for representing conversations as DAGs for visualization, analysis, and pre-processing.

* Free software: MIT license

Installation
============

::

    pip install pyconversations

You can also install the in-development version with::

    pip install https://github.com/hunter-heidenreich/pyconversations/archive/master.zip


Documentation
=============


https://pyconversations.readthedocs.io/


Development
===========

To run all the tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
