# -*- coding: utf-8 -*-
from ..base import Base


class RunManager(Base):
    """Run Manager base class

    Builds and runs an experiment based on an experiment configuration file.
    Optionally calculates metrics based on the output of the experiment.
    """
