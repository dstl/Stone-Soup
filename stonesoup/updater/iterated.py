# -*- coding: utf-8 -*-

from .base import Updater
from ..base import Property

class IteratedUpdater(Updater):
    r"""Slightly different from your usual updater in that it takes an updater input and runs
    :meth:`update()` as an iteration over that update."""

    updater = Property(Updater, doc="")
    tolerance = Property(float, default=1e-8, doc="")
