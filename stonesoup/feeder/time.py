# -*- coding: utf-8 -*-
import bisect
import datetime
from warnings import warn

from ..base import Property
from .base import Feeder


class TimeBufferedFeeder(Feeder):
    """Buffer detections so they can be yielded in time order.

    Any "old" detections (where the time is earlier than the head of the
    buffer) shall be dropped, producing a :class:`UserWarning`.
    """
    buffer_size = Property(int, default=1000, doc="Max size of buffer")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._detections = set()

    @property
    def detections(self):
        return self._detections

    def detections_gen(self):
        detections_iter = iter(self.detector.detections_gen())
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
                time, detections = time_detections_buffer.pop(0)
                self._detections = detections
                yield time, detections

        # No more new detections: yield remaining buffer
        for time, detections in time_detections_buffer:
            self._detections = detections
            yield time, detections


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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._detections = set()

    @property
    def detections(self):
        return self._detections

    def detections_gen(self):
        detections_iter = iter(self.detector.detections_gen())
        prev_time, detections = next(detections_iter)
        detections_buffer = set(detections)

        for time, detections in detections_iter:
            if time > prev_time + self.time_window:
                self._detections = detections_buffer
                yield prev_time + self.time_window, self.detections

                # Increment time and yield empty detections set until latest
                # detections within time window.
                prev_time += self.time_window
                while time > prev_time + self.time_window:
                    self._detections = set()
                    prev_time += self.time_window
                    yield prev_time, self.detections

                detections_buffer = set(detections)
            else:
                detections_buffer.update(detections)

        # No more new detections: yield remaining buffer
        self._detections = detections_buffer
        yield prev_time + self.time_window, self.detections
