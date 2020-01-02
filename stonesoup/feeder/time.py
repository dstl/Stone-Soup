# -*- coding: utf-8 -*-
import bisect
import datetime
from warnings import warn

from ..base import Property
from ..buffered_generator import BufferedGenerator
from .base import Feeder


class TimeBufferedFeeder(Feeder):
    """Buffer detections so they can be yielded in time order.

    Any "old" detections (where the time is earlier than the head of the
    buffer) shall be dropped, producing a :class:`UserWarning`.
    """
    buffer_size = Property(int, default=1000, doc="Max size of buffer")

    @BufferedGenerator.generator_method
    def detections_gen(self):
        detections_iter = iter(self.detector)
        time_detections_buffer = [next(detections_iter)]

        for time_detections in detections_iter:
            # Drop "old" detections
            if len(time_detections_buffer) >= self.buffer_size and \
                    time_detections < time_detections_buffer[0]:
                warn('"Old" detection dropped')
                continue

            # Insert in order
            bisect.insort(time_detections_buffer, time_detections)

            # Yield oldest when buffer full
            if len(time_detections_buffer) > self.buffer_size:
                yield time_detections_buffer.pop(0)

        # No more new detections: yield remaining buffer
        yield from time_detections_buffer


class TimeSyncFeeder(Feeder):
    """Synchronise the detections into selected time window.

    Assumes that detections from :attr:`detector` are in time order. The
    :class:`~.TimeBufferedFeeder` can be used in conjunction to ensure this is
    the case.
    """

    time_window = Property(
        datetime.timedelta,
        default=datetime.timedelta(seconds=1),
        doc="Time window to group detections")

    @BufferedGenerator.generator_method
    def detections_gen(self):
        detections_iter = iter(self.detector)
        prev_time, detections = next(detections_iter)
        detections_buffer = set(detections)

        prev_time -= self.time_window

        for time, detections in detections_iter:
            if time > prev_time + self.time_window:
                yield prev_time + self.time_window, detections_buffer

                # Increment time and yield empty detections set until latest
                # detections within time window.
                prev_time += self.time_window
                while time > prev_time + self.time_window:
                    prev_time += self.time_window
                    yield prev_time, set()

                detections_buffer = set(detections)
            else:
                detections_buffer.update(detections)

        # No more new detections: yield remaining buffer
        yield prev_time + self.time_window, detections_buffer
