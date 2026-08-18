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

Optional dependencies
^^^^^^^^^^^^^^^^^^^^^
Some Stone Soup features use dependencies that are not required by the core
package. These can be installed with ``pip`` extras. Multiple extras can be
installed together, for example:

.. code::

    python -m pip install "stonesoup[video,optuna]"

The currently available feature extras are:

.. list-table::
    :header-rows: 1
    :widths: 20 50 30

    * - Extra
      - Feature
      - Optional dependencies
    * - ``video``
      - Video reading, processing and related demonstrations
      - ffmpeg-python, moviepy, OpenCV, yt-dlp
    * - ``tensorflow``
      - TensorFlow integrations
      - TensorFlow
    * - ``ultralytics``
      - Ultralytics object detection
      - ultralytics
    * - ``mfa``
      - Multi-frame assignment data association
      - OR-Tools
    * - ``ehm``
      - Efficient hypothesis management data association
      - pyehm
    * - ``optuna``
      - Optuna-based sensor management
      - optuna
    * - ``ode``
      - ODE-based functionality and examples
      - PyTorch
    * - ``roadnet``
      - Road-network functionality
      - GeoPandas, NetworkX
    * - ``architectures``
      - Architecture graph visualisation
      - Graphviz, NetworkX, pydot

The ``dev`` extra installs the dependencies used for development and testing,
and the ``docs`` extra installs additional dependencies required when building
the documentation. The authoritative list of extras and packages is maintained
in ``pyproject.toml``.


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
    auto_tutorials/index
    auto_examples/index
    auto_demos/index
    contributing
    copyright

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`

