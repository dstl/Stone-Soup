# -*- coding: utf-8 -*-
import re
from collections.abc import Sequence

from stonesoup.base import Base

STONESOUP_TYPE_REGEX = re.compile(r'stonesoup\.(\w+\.)*')


def _headings(heading, lines):
    try:
        index = lines.index(heading) + 2
    except ValueError:
        # No placeholder found, so extend.
        # Numpydoc will ignore empty list: no need to check for contents
        if lines and not lines[-1] == "":
            lines.append("")
        lines.extend([heading, "-" * len(heading)])
        index = len(lines)

    return index


def declarative_class(app, what, name, obj, options, lines):
    """Add declared properties to Parameters list for numpydoc"""
    if what == "class" and issubclass(obj, Base):
        param_index = _headings("Parameters", lines)
        attr_index = _headings("Attributes", lines)
        for name, property_ in obj.properties.items():
            # there may be a better way to do the check below, but the typing API is variable
            # across Python versions, making it tricky and this may do well enough.
            if hasattr(property_.cls, '__module__') and property_.cls.__module__ == 'typing':
                class_name = str(property_.cls)
                class_name = class_name.replace('typing.', '')
                class_name = STONESOUP_TYPE_REGEX.sub('', class_name)
                is_sequence = False
            else:
                is_sequence = isinstance(property_.cls, Sequence)
                if is_sequence:
                    cls = property_.cls[0]
                else:
                    cls = property_.cls
                module_name = cls.__module__
                cls_name = cls.__name__
                class_name = "{}.{}".format(
                    module_name, cls_name)
            # To shorten names for builtins and also stonesoup components
            tild = class_name.split(".")[0] in ("stonesoup", "builtins")
            # To add optional if default value is defined.
            is_optional = property_.default is not property_.empty
            doc_type = "{}:class:`{}{}`{}".format(
                "sequence of " if is_sequence else "",
                "~" if tild else "",
                class_name,
                ", optional" if is_optional else "",
            )

            new_lines = "{} : {}\n    {}".format(
                name, doc_type, property_.doc or "").split("\n")
            lines[param_index:param_index] = new_lines
            param_index += len(new_lines)
            attr_index += len(new_lines)
            lines.insert(attr_index, name)
            attr_index += 1


def setup(app):
    app.connect('autodoc-process-docstring', declarative_class)


import os
import matplotlib
import matplotlib.pyplot as plt

from sphinx_gallery.scrapers import figure_rst


class gallery_scraper():
    def __init__(self):
        self.plotted_figures = set()

    def __call__(self, block, block_vars, gallery_conf, **kwargs):
        """Scrape Matplotlib images.

        Parameters
        ----------
        block : tuple
            A tuple containing the (label, content, line_number) of the block.
        block_vars : dict
            Dict of block variables.
        gallery_conf : dict
            Contains the configuration of Sphinx-Gallery
        **kwargs : dict
            Additional keyword arguments to pass to
            :meth:`~matplotlib.figure.Figure.savefig`, e.g. ``format='svg'``.
            The ``format`` kwarg in particular is used to set the file extension
            of the output file (currently only 'png', 'jpg', and 'svg' are
            supported).

        Returns
        -------
        rst : str
            The ReSTructuredText that will be rendered to HTML containing
            the images. This is often produced by :func:`figure_rst`.
        """

        from matplotlib.figure import Figure

        image_path_iterator = block_vars['image_path_iterator']
        image_paths = list()
        new_figures = set(plt.get_fignums()) - self.plotted_figures
        last_line = block[1].strip().split('\n')[-1]
        output = block_vars['example_globals'].get(last_line)
        if isinstance(output, Figure):
            new_figures.add(output.number)

        for fig_num, image_path in zip(new_figures, image_path_iterator):
            if 'format' in kwargs:
                image_path = '%s.%s' % (os.path.splitext(image_path)[0],
                                        kwargs['format'])
            # Set the fig_num figure as the current figure as we can't
            # save a figure that's not the current figure.
            fig = plt.figure(fig_num)
            self.plotted_figures.add(fig_num)
            to_rgba = matplotlib.colors.colorConverter.to_rgba
            # shallow copy should be fine here, just want to avoid changing
            # "kwargs" for subsequent figures processed by the loop
            these_kwargs = kwargs.copy()
            for attr in ['facecolor', 'edgecolor']:
                fig_attr = getattr(fig, 'get_' + attr)()
                default_attr = matplotlib.rcParams['figure.' + attr]
                if to_rgba(fig_attr) != to_rgba(default_attr) and \
                        attr not in kwargs:
                    these_kwargs[attr] = fig_attr
            fig.savefig(image_path, **these_kwargs)
            image_paths.append(image_path)
        return figure_rst(image_paths, gallery_conf['src_dir'])
