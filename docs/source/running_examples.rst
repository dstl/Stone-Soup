Running examples outside notebooks
==================================

Stone Soup's tutorials, examples and demonstrations are built primarily for
Sphinx-Gallery and notebook-style use. When the same code is run as a normal
Python script or from an IDE, the plotting backend may not display the final
figure automatically. The plotting object is still created; it just needs an
explicit display step appropriate to the backend.

Matplotlib ``Plotter``
----------------------

For :class:`~.Plotter`, display the completed figure with Matplotlib's
``show`` function at the end of the script::

    from matplotlib import pyplot as plt
    from stonesoup.plotter import Plotter

    plotter = Plotter()
    # Add truths, measurements, tracks, etc.
    plt.show()

This keeps interactive display outside the plotting component itself and lets
notebooks and Sphinx-Gallery continue to manage rendering in their normal way.

Plotly ``Plotterly``
--------------------

For :class:`~.Plotterly`, call ``show`` on the underlying Plotly figure::

    from stonesoup.plotter import Plotterly

    plotter = Plotterly()
    # Add truths, measurements, tracks, etc.
    plotter.fig.show()

Plotly chooses a renderer according to the environment. If an IDE does not
open the figure with its default renderer, select a suitable renderer before
calling ``show``. For example, to open the result in the system browser::

    import plotly.io as pio

    pio.renderers.default = "browser"
    plotter.fig.show()

For environments where interactive renderer startup is unreliable, writing a
standalone HTML file provides a deterministic alternative::

    plotter.fig.write_html("stone_soup_plot.html", auto_open=False)

The generated file can then be opened directly in a browser and does not need
a running Python process after it has been written.

Sphinx-Gallery and notebooks
----------------------------

The explicit display calls above are intended for scripts and IDE execution.
They do not need to be added to Stone Soup's documentation examples merely to
make the rendered documentation display a figure. Sphinx-Gallery and notebook
frontends already capture the plotting objects as part of their own rendering
process.

If a Plotly figure still fails to appear in a particular IDE, check that IDE's
Plotly renderer support and use ``write_html`` as the portable fallback.
