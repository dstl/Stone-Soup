# -*- coding: utf-8 -*-
from collections.abc import Sequence

from stonesoup.base import Base


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
            is_sequence = isinstance(property_.cls, Sequence)
            if is_sequence:
                cls = property_.cls[0]
            else:
                cls = property_.cls
            class_name = "{}.{}".format(
                cls.__module__, cls.__name__)
            # To shorten names for builtins and also stonesoup components
            tild = class_name.split(".")[0] in ("stonesoup", "builtins")
            # To add optional if default value is defined.
            is_optional = property_.default is not property_.empty
            doc_type = "{}:class:`{}{}`{}".format(
                is_sequence and "sequence of " or "",
                tild and "~" or "",
                class_name,
                is_optional and ", optional" or "",
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
