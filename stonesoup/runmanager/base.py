# -*- coding: utf-8 -*-
from ..base import Base, Property


class RunManager(Base):
    """Run Manager base class

    Builds and runs an experiment based on an experiment configuration file.
    Optionally calculates metrics based on the output of the experiment.
    """

    #results_tracks = Property(
    #    Tracker, doc="Tracks which metrics will be generated for")
