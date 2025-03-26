import heapq
from abc import abstractmethod
from collections.abc import Collection
from functools import cached_property
from queue import Empty, Full, Queue, LifoQueue, PriorityQueue
from threading import Thread

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


class _QueueMultiDataFeeder(DetectionFeeder, GroundTruthFeeder):
    reader = None
    readers: Collection[Reader] = Property(doc='Readers to yield from')
    max_size: int = Property(
        default=0,
        doc="Max queue size, where it will block more data being added. Default 0, unbounded.")

    @cached_property
    @abstractmethod
    def _queue(self):
        raise NotImplementedError

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._threads = None

    @staticmethod
    def _run(queue: Queue, reader: Reader):
        for time, data in reader:
            queue.put((time, data))

    @BufferedGenerator.generator_method
    def data_gen(self):
        if self._threads is None:
            self._threads = [
                Thread(target=self._run, args=(self._queue, reader), daemon=True)
                for reader in self.readers
            ]
            for thread in self._threads:
                thread.start()

        while any(thread.is_alive() for thread in self._threads) \
                or not self._queue.empty():
            try:
                time, data = self._queue.get_nowait()
            except Empty:
                continue
            yield time, data
            self._queue.task_done()


class FIFOMultiDataFeeder(_QueueMultiDataFeeder):
    """FIFO Multi-data Feeder

    This returns data from multiple data readers as a single stream,
    where each reader is consumed in a separate thread and put into a
    queue. The data is consumed first in, first out.

    This is aimed at sources of data that are real-time streams, for
    example sensors sending data via network.
    """

    @cached_property
    def _queue(self):
        return Queue(self.max_size)


class LIFOMultiDataFeeder(_QueueMultiDataFeeder):
    """LIFO Multi-data Feeder

    This returns data from multiple data readers as a single stream,
    where each reader is consumed in a separate thread and put into a
    queue. The data is consumed last in, first out.

    This is aimed at sources of data that are real-time streams, for
    example sensors sending data via network.
    """

    @cached_property
    def _queue(self):
        return LifoQueue(self.max_size)


class PriorityMultiDataFeeder(_QueueMultiDataFeeder):
    """Priority Multi-data Feeder

    This returns data from multiple data readers as a single stream,
    where each reader is consumed in a separate thread and put into a
    queue. The data is consumed prioritised by time, earlier to later.

    This is aimed at sources of data that are real-time streams, for
    example sensors sending data via network.
    """

    @cached_property
    def _queue(self):
        return PriorityQueue(self.max_size)


class _MaxSizePriorityQueue(PriorityQueue):
    def put(self, item, block=True, timeout=None):
        try:
            # Call super, so can at least try or wait timeout before
            # overwriting.
            super().put(item, False, timeout)
        except Full:
            with self.not_full:
                self._put(item)
                self.unfinished_tasks += 1
                self.not_empty.notify()

    def _put(self, item):
        if self.maxsize <= 0 or self.maxsize > len(self.queue):
            heapq.heappush(self.queue, item)
        else:
            heapq.heappushpop(self.queue, item)


class MaxSizePriorityMultiDataFeeder(_QueueMultiDataFeeder):
    """Max Size Priority Multi-data Feeder

    This returns data from multiple data readers as a single stream,
    where each reader is consumed in a separate thread and put into a
    queue. The data is consumed prioritised by time, earlier to later.

    Unlike :class:`~.PriorityMultiDataFeeder`, rather than blocking
    when the queue is full, it will drop the oldest data.

    This is aimed at sources of data that are real-time streams, for
    example sensors sending data via network.
    """
    max_size: int = Property(
        default=0,
        doc="Max queue size, where it will drop oldest data when full. Default 0, unbounded.")

    @cached_property
    def _queue(self):
        return _MaxSizePriorityQueue(self.max_size)
