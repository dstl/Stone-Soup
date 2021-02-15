"""
Plugin system for Stone Soup.

Stone Soup is able to import plugins using package metadata. Packages can
register themselves for discovery by providing the :attr:`entry_points` argument
to setup() in :attr:`setup.py`.

For example if you have a package named :attr:`my_package` and its :attr:`setup.py` file includes:

.. code-block:: python

    setup(
        ...
        entry_points={'stonesoup.plugins': 'my_plugin = my_package'}
        ...
    )

Then Stone Soup will discover your plugin and load all of the registered entry points. It is
possible to name your plugin the same name as your package name in :attr:`setup()`. Your plugin
can be loaded using:

.. code-block:: python

    from stonesoup.plugins.my_plugin import MyClass

.. note::
    When developing plugins for Stone Soup, :attr:`entry_points` must be associated with
    the :attr:`stonesoup.plugins` key in the :attr:`entry_points` dictionary.
"""
import sys
import pkg_resources
import warnings
import importlib.util

for entry_point in pkg_resources.iter_entry_points('stonesoup.plugins'):
    try:
        name = entry_point.name
        plugin_module = f'{__name__}.{name}'
        sys.modules[plugin_module] = entry_point.load()

    except (ImportError, ModuleNotFoundError) as e:
        warnings.warn(f'Failed to load module. {e}')
