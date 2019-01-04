# -*- coding: utf-8 -*-
from stonesoup.base import Base, ListProperty


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
            is_list = isinstance(property_, ListProperty)
            class_name = "{}.{}".format(
                property_.cls.__module__, property_.cls.__name__)
            # To shorten names for builtins and also stonesoup components
            tild = class_name.split(".")[0] in ("stonesoup", "builtins")
            # To add optional if default value is defined.
            is_optional = property_.default is not property_.empty
            doc_type = "{}:class:`{}{}`{}".format(
                is_list and "list of " or "",
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
