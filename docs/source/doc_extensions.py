# -*- coding: utf-8 -*-
from stonesoup.base import Base


def declarative_class(app, what, name, obj, options, lines):
    """Add declared properties to Parameters list for numpydoc"""
    if what == "class" and issubclass(obj, Base):
        try:
            param_index = lines.index("Parameters") + 2
        except ValueError:
            # No placeholder found, so extend.
            # Numpydoc will ignore empty list: no need to check for properties
            if lines and not lines[-1] == "":
                lines.append("")
            lines.extend(["Parameters", "----------"])
            param_index = len(lines)

        for name, property_ in obj.properties.items():
            class_name = "{}.{}".format(
                property_.cls.__module__, property_.cls.__name__)
            # To shorten names for builtins and also stonesoup components
            tild = class_name.split(".")[0] in ("stonesoup", "builtins")
            doc_type = ":class:`{}{}`".format(tild and "~" or "", class_name)
            # Add optional if default value is defined.
            if property_.default is not property_.empty:
                doc_type += ", optional"
            new_lines = "{} : {}\n    {}".format(
                name, doc_type, property_.doc or "").split("\n")
            lines[param_index:param_index] = new_lines
            param_index += len(new_lines)


def setup(app):
    app.connect('autodoc-process-docstring', declarative_class)
