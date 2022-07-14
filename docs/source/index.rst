Welcome to Stone Soup's documentation!
======================================

.. image:: _static/stone_soup_logo.svg
    :width: 200px
    :align: center
    :alt: Stone Soup Logo

Stone Soup is a software project to provide the target tracking and state
estimation community with a framework for the development and testing of
tracking and state estimation algorithms.

As Stone Soup is focused on development and testing of algorithms, and such
components may not be the most optimised implementations, instead focusing on
being flexible. Its also intended to aid choice of component/algorithms
to tackle real world problems.

Stone Soup is under active development, where feedback and contributions are
welcomed to grow the number of components and features available.

Please see the Stone Soup :ref:`auto_tutorials/index:Tutorials` for learning
about tracking and using Stone Soup, :ref:`auto_examples/index:Examples` for
examples of Stone Soup features, and :ref:`auto_demos/index:Demonstrations`
for demonstrations of using Stone Soup.

For community support, head over to the
`Stone Soup room on Gitter <https://gitter.im/dstl/Stone-Soup>`_.

Installation
------------
To install Stone Soup from PyPI with ``pip``:

.. code::

    python -m pip install stonesoup

To install Stone Soup from Conda-Forge with ``conda``:

.. code::

    conda config --add channels conda-forge
    conda install stonesoup

Stone Soup is currently in active development. To install
the latest version from the GitHub repository:

.. code::

    python -m pip install git+https://github.com/dstl/Stone-Soup.git#egg=stonesoup


Developing
^^^^^^^^^^
If you are looking to carry out development with Stone Soup, you should first
clone from GitHub and install with development dependencies by doing the
following:

.. code::

    git clone "https://github.com/dstl/Stone-Soup.git"
    cd Stone-Soup
    python -m pip install -e ".[dev]"

Please also see our :ref:`contributing:Contributing` page.

Contents
========

.. toctree::
    :maxdepth: 2

    design
    stonesoup
    runmanager
    auto_tutorials/index
    auto_examples/index
    auto_demos/index
    contributing
    copyright

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`

