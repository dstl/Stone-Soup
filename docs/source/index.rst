Welcome to Stone Soup's documentation!
======================================

.. image:: _static/stone_soup_logo.png
    :scale: 80%
    :align: center
    :alt: Stone Soup Logo

Stone Soup is a software project to provide the target tracking community with
a framework for the development and testing of tracking algorithms.

Installation
------------
To install Stone Soup from PyPI execute:

.. code::

    python -m pip install stonesoup

If you are looking to carry out development with Stone Soup, you should first
clone from GitHub and install with development dependencies by doing the
following:

.. code::

    git clone "https://github.com/dstl/Stone-Soup.git"
    cd Stone-Soup
    python -m pip install -e .[dev]


Contents:

.. toctree::
    :maxdepth: 2

    dataflow
    interface
    types
    stonesoup.config
    stonesoup.base
    stonesoup.functions
    stonesoup.measures
    stonesoup.serialise
    stonesoup
    contributing
    copyright

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`

