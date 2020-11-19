# -*- coding: utf-8 -*-
import heapq
from typing import Collection

from .base import DetectionFeeder, GroundTruthFeeder
from ..base import Property
from ..buffered_generator import BufferedGenerator
from ..reader import Reader


class MultiDataFeeder(DetectionFeeder, GroundTruthFeeder):
    """Multi-data Feeder

    This returns states from multiple data readers as a single stream,
    yielding from the reader yielding the lowest timestamp first.
    """
    reader = None
    readers: Collection[Reader] = Property(doc='Readers to yield from')

    @BufferedGenerator.generator_method
    def data_gen(self):
        yield from heapq.merge(*self.readers)
