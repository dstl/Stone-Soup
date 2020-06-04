Contributing
============

Code Style
----------
* Use clear naming for variables, objects, etc. rather than mathematical short
  hand (e.g. `measurement_matrix` rather than `H`)
* Use standard Python errors and warnings
* At least one release cycle with deprecated warnings before removing/changing
  interface (excluding alpha/beta releases)
* No strict typing, exploit duck typing
* Each object has one role, modular sub-components
* Embrace abstract base classes
* Allowing inhomogeneity at the lower levels of the inheritance hierarchy
* New components should subclass from base types for each (e.g. Updater) where
  possible
* Use of `Python abstract classes`_ to have familiar behaviour to existing
  Python objects
* Avoid strict encapsulation (use of `__methods`) but use non-strict
  encapsulation where needed (`_method`).
* All code should follow :pep:`8`. Flake8_ can be used to check for issues. If
  necessary, in-line ignores can be added.

Documentation
-------------
In Stone Soup, `NumPy Doc`_ style documentation is used, with documentation
generated with Sphinx. It must be provided for all public interfaces, and
should also be provided for private interfaces.

Where applicable, documentation should include reference to the paper and
mathematical description.
For new functionality provide `sphinx-gallery` example which demonstrate use
case.

Tests
-----
PyTest_ is used for testing in Stone Soup. As much effort should be put into
developing tests as the code. Tests should be provide to test functionality and
also ensuring exceptions are raised or managed appropriately.

License
-------
Any contributions submitted are to be under the MIT_ or similar non-copyleft
license. MIT_ License will be assumed to be unless otherwise stated on the pull
request.

External Dependencies
---------------------
Use standard library and existing well maintained external libraries where
possible. New external libraries should be licensed permissive (e.g MIT_) or
weak copyleft (e.g. LGPL_)

Pull Requests
-------------
Submissions should be done via Pull Requests on the `Stone Soup GitHub repo`_.
Currently we are using `GitHub Flow`_  as our approach to development. Once a
pull request has been opened, CircleCI_ will run tests and build documentation.

.. _NumPy Doc: https://numpydoc.readthedocs.io/en/latest/format.html
.. _Flake8: https://flake8.pycqa.org/en/latest/
.. _Python abstract classes: https://docs.python.org/3/library/abc.html
.. _PyTest: https://docs.pytest.org/en/latest/
.. _MIT: https://opensource.org/licenses/MIT
.. _LGPL: https://opensource.org/licenses/lgpl-license
.. _Stone Soup GitHub repo: https://github.com/dstl/Stone-Soup
.. _GitHub Flow: https://guides.github.com/introduction/flow/index.html
.. _CircleCI: https://circleci.com/
