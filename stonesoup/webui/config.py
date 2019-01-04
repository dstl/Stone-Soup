import inspect
import logging
import uuid
from itertools import chain
from numbers import Number

from flask import render_template, request
from flask_socketio import SocketIO
from jinja2 import Undefined
import numpy as np

from .construct import classes, construct, ConstructError
from ..base import Base, Property, ListProperty
from ..types import Type
from ..serialise import YAML

from ..webui import app, logger

socketio = SocketIO(app)
app.add_template_test(inspect.isabstract, 'abstract')
app.add_template_test(lambda class_: issubclass(class_, Type), 'type')
app.add_template_test(lambda class_: issubclass(class_, Number), 'numbertype')
app.add_template_test(
    lambda object_: isinstance(object_, np.ndarray), 'ndarray')
app.add_template_test(lambda object_: object_ is Property.empty, 'empty')
app.add_template_test(
    lambda class_: issubclass(class_, Base) and not issubclass(class_, Type),
    'component')
app.add_template_test(
    lambda property_: isinstance(property_, ListProperty), 'listproperty')
app.jinja_env.globals['uuid'] = uuid
app.jinja_env.globals['id'] = id
app.jinja_env.globals['chain'] = chain
app.jinja_env.globals['components'] = Base.subclasses


@app.template_test()
def subclass(class_, classinfo=None):
    if isinstance(classinfo, Undefined):
        classinfo = object
    return issubclass(class_, classinfo)


@app.template_global()
def get_class_macro(macros, class_):
    """Fetch the first matching macro, work through classes."""
    for name in (cls.__qualname__.split('.')[-1] for cls in class_.__mro__):
        if hasattr(macros, name):
            return getattr(macros, name)
    raise ValueError("No macro found")


@socketio.on_error_default
def error_handler(e):
    logger.error(e)
    raise e


class SocketIOHandler(logging.Handler):
    def emit(self, record):
        # TODO: Html template for log message?
        socketio.send((record.levelno, record.getMessage()))


sio = SocketIOHandler()
logger.addHandler(sio)


@app.route("/")
def index():
    return render_template('index.html', )


@socketio.on('connect')
def connect():
    logger.info("New connection %s", request.sid)


@socketio.on('get_component_form')
def get_component_form(component_class_name, component_name_prefix):
    logger.debug("Form for %r requested for %r",
                 component_class_name, component_name_prefix)
    return render_template("config.html",
                           component=classes[component_class_name],
                           prefix=component_name_prefix,
                           instance=None,
                           )


@socketio.on('submit_config')
def submit_config(config_map):
    yaml = YAML()
    logger.debug("Config submitted %r", config_map)
    try:
        component = construct(config_map['config']['tracker'])
    except ConstructError as err:
        logging.error(err)
        logging.debug(err, exc_info=True)
        return {'status': "error", 'message': str(err.error),
                'name': "config[tracker][{}]".format("][".join(err.keys))}

    config_yml = yaml.dumps(component)
    logging.debug("Config generated: %r", config_yml)

    return {'status': "success", 'config': config_yml}


@socketio.on("load_config")
def load_config(config_str):
    yaml = YAML()
    component = yaml.load(config_str)
    return render_template("config.html",
                           component=component.__class__,
                           prefix="config[tracker]",
                           instance=component,
                           )
