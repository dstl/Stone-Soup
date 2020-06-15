# -*- coding: utf-8 -*-
import datetime
import heapq
from warnings import warn

from ..base import Property
from ..buffered_generator import BufferedGenerator
from .base import DetectionFeeder, GroundTruthFeeder


class TimeBufferedFeeder(DetectionFeeder, GroundTruthFeeder):
    """Buffer data so it can be yielded in time order.

    Any "old" data (where the time is earlier than the head of the
    buffer) shall be dropped, producing a :class:`UserWarning`.
    """
    buffer_size = Property(int, default=1000, doc="Max size of buffer")

    @BufferedGenerator.generator_method
    def data_gen(self):
        time_data_buffer = []

        for time_data in self.reader:
            # Drop "old" detections
            if len(time_data_buffer) >= self.buffer_size and \
                    time_data < time_data_buffer[0]:
                warn('"Old" detection dropped')
                continue

            # Yield oldest when buffer full
            if len(time_data_buffer) >= self.buffer_size:
                yield heapq.heappushpop(time_data_buffer, time_data)
            else:
                # Else just insert
                heapq.heappush(time_data_buffer, time_data)

        # No more new data: yield remaining buffer
        while time_data_buffer:
            yield heapq.heappop(time_data_buffer)


class TimeSyncFeeder(DetectionFeeder, GroundTruthFeeder):
    """Synchronise the data into selected time window.

    Assumes that states from :attr:`Reader` are in time order. The
    :class:`~.TimeBufferedFeeder` can be used in conjunction to ensure this is
    the case.
    """

    time_window = Property(
        datetime.timedelta,
        default=datetime.timedelta(seconds=1),
        doc="Time window to group detections")

    @BufferedGenerator.generator_method
    def data_gen(self):
        data_iter = iter(self.reader)
        prev_time, states = next(data_iter)
        data_buffer = set(states)

        prev_time -= self.time_window

        for time, states in data_iter:
            if time > prev_time + self.time_window:
                yield prev_time + self.time_window, data_buffer

                # Increment time and yield empty data set until latest
                # state within time window.
                prev_time += self.time_window
                while time > prev_time + self.time_window:
                    prev_time += self.time_window
                    yield prev_time, set()

                data_buffer = set(states)
            else:
                data_buffer.update(states)

        # No more new states: yield remaining buffer
        yield prev_time + self.time_window, data_buffer
