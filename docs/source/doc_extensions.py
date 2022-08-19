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


def shorten_type_hints(app, what, name, obj, options, signature, return_annotation):
    if signature is not None:
        signature = STONESOUP_TYPE_REGEX.sub('', signature)
    return signature, return_annotation


def setup(app):
    app.connect('autodoc-process-docstring', declarative_class)
    app.connect('autodoc-process-signature', shorten_type_hints)


import os
import matplotlib
import matplotlib.pyplot as plt
from textwrap import indent

from sphinx_gallery.scrapers import (
    figure_rst, _anim_rst, _matplotlib_fig_titles, HLIST_HEADER,
    HLIST_IMAGE_MATPLOTLIB)

import plotly.graph_objects as go
try:
    import kaleido
except ImportError:
    write_plotly_image = None
else:
    from plotly.io import write_image as write_plotly_image


class gallery_scraper():
    def __init__(self):
        self.plotted_figures = set()
        self.current_src_file = None

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
        # New file, so close all currently open figures
        if block_vars['src_file'] != self.current_src_file:
            for fig in self.plotted_figures:
                plt.close(fig)
            self.plotted_figures = set()
            self.current_src_file = block_vars['src_file']

        from matplotlib.animation import Animation
        from matplotlib.figure import Figure
        image_path_iterator = block_vars['image_path_iterator']
        image_rsts = []

        # Check for animations
        anims = list()
        if gallery_conf.get('matplotlib_animations', False):
            for ani in block_vars['example_globals'].values():
                if isinstance(ani, Animation):
                    anims.append(ani)

        # Then standard images
        new_figures = set(plt.get_fignums()) - self.plotted_figures
        last_line = block[1].strip().split('\n')[-1]
        variable, *attributes = last_line.split(".")
        try:
            output = block_vars['example_globals'][variable]
            for attribute in attributes:
                output = getattr(output, attribute)
        except (KeyError, AttributeError):
            pass
        else:
            if isinstance(output, Figure):
                new_figures.add(output.number)
            elif isinstance(output, go.Figure):
                if write_plotly_image is not None:
                    image_path = next(image_path_iterator)
                    if 'format' in kwargs:
                        image_path = '%s.%s' % (os.path.splitext(image_path)[0],
                                                kwargs['format'])
                    write_plotly_image(output, image_path, kwargs.get('format'))

        for fig_num, image_path in zip(new_figures, image_path_iterator):
            if 'format' in kwargs:
                image_path = '%s.%s' % (os.path.splitext(image_path)[0],
                                        kwargs['format'])
            # Set the fig_num figure as the current figure as we can't
            # save a figure that's not the current figure.
            fig = plt.figure(fig_num)
            self.plotted_figures.add(fig_num)
            # Deal with animations
            cont = False
            for anim in anims:
                if anim._fig is fig:
                    image_rsts.append(_anim_rst(anim, image_path, gallery_conf))
                    cont = True
                    break
            if cont:
                continue
            # get fig titles
            fig_titles = _matplotlib_fig_titles(fig)
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
            these_kwargs['bbox_inches'] = "tight"
            fig.savefig(image_path, **these_kwargs)
            image_rsts.append(
                figure_rst([image_path], gallery_conf['src_dir'], fig_titles))
        rst = ''
        if len(image_rsts) == 1:
            rst = image_rsts[0]
        elif len(image_rsts) > 1:
            image_rsts = [re.sub(r':class: sphx-glr-single-img',
                                 ':class: sphx-glr-multi-img',
                                 image) for image in image_rsts]
            image_rsts = [HLIST_IMAGE_MATPLOTLIB + indent(image, u' ' * 6)
                          for image in image_rsts]
            rst = HLIST_HEADER + ''.join(image_rsts)
        return rst


class reset_numpy_random_seed:

    def __init__(self):
        self.state = None

    def __call__(self, gallery_conf, fname, when):
        import numpy as np
        if when == 'before':
            self.state = np.random.get_state()
        elif when == 'after':
            # Set state attribute back to `None`
            self.state = np.random.set_state(self.state)
