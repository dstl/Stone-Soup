# -*- coding: utf-8 -*-
from typing import Union

from ..base import Property
from ..hypothesiser import Hypothesiser


class Gater(Hypothesiser):
    """Gater base class

    Gaters wrap :class:`.Hypothesiser` objects and can be used to modify (typically reduce) the
    returned hypotheses.
    """

    hypothesiser: Union[Hypothesiser, 'Gater'] = Property(
        doc="Hypothesiser or Gater that is being wrapped.")
