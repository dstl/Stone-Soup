# -*- coding: utf-8 -*-
from ..base import Property
from ..hypothesiser import Hypothesiser


class Gater(Hypothesiser):
    """Gater base class

    Gaters wrap :class:`.Hypothesiser` objects and can be used to modify (typically reduce) the
    returned hypotheses.
    """

    hypothesiser = Property(
        Hypothesiser, doc="Hypothesiser that is being wrapped.")
