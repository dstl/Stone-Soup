# -*- coding: utf-8 -*-
from stonesoup.base import Base


def declarative_signature(
        app, what, name, obj, options, signature, return_annotation):
    if issubclass(obj, Base):
        args = []
        for name, property_ in obj.properties.items():
            try:
                args.append("{}={!r}".format(name, property_.default))
            except AttributeError:
                args.append(name)
        signature = "({})".format(",".join(args))
        return signature, return_annotation


def setup(app):
    app.connect('autodoc-process-signature', declarative_signature)
