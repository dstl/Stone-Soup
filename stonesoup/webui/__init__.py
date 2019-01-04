# -*- coding: utf-8 -*-
import logging
import tempfile

from flask import Flask
from flask_caching import Cache

logger = logging.getLogger(__name__)
app = Flask(__name__)

cache = Cache(app, config={
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': tempfile.TemporaryDirectory(prefix="soup_webui_").name
})
