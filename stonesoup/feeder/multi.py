# -*- coding: utf-8 -*-
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

    readers = Property([Reader], doc='Readers to yield from')

    @BufferedGenerator.generator_method
    def data_gen(self):
        reader_iters = (iter(reader) for reader in self.readers)
        iter_data = {
            reader_iter: next(reader_iter)
            for reader_iter in reader_iters}

        min_time = None
        while iter_data:  # Whilst still iterators left
            for reader_iter, (time, data) in iter_data.items():
                if min_time is None or time < min_time:
                    min_time = time
                    min_reader_iter = reader_iter
                    min_data = data

            yield min_time, min_data
            min_time = None

            try:
                # Grab next set for this iter
                iter_data[min_reader_iter] = next(min_reader_iter)
            except StopIteration:
                # Empty iterator, remove from dictionary
                del iter_data[min_reader_iter]
