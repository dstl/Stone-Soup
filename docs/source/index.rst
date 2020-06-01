Welcome to Stone Soup's documentation!
======================================

.. image:: _static/stone_soup_logo.png
    :scale: 80%
    :align: center
    :alt: Stone Soup Logo

Stone Soup is a software project to provide the target tracking and state
estimation community with a framework for the development and testing of
tracking and state estimation algorithms.

As Stone Soup is focused on development and testing of algorithms, and such
components may not be the most optimised implementations, instead focusing on
being flexible. Its also intended to aid choice of component/algorithms
to tackle real world problems.

Stone Soup is currently in *beta* and under continuing development, where
feedback and contributions are welcomed to improve the component interface
design and grow the number components available.

Please see the example
`Stone Soup Notebooks <https://github.com/dstl/Stone-Soup-Notebooks>`_.

For community support, head over to the
`Stone Soup room on Gitter <https://gitter.im/dstl/Stone-Soup>`_.

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
    stonesoup
    contributing
    copyright
    auto_tutorials/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`

