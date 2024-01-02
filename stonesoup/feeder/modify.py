import copy
from abc import abstractmethod
from datetime import datetime, timedelta
from typing import Tuple

from .base import Feeder
from .simple import IterFeeder
from ..base import Property
from ..buffered_generator import BufferedGenerator


class BaseModifiedFeederIter(IterFeeder):
    def __iter__(self):
        self.reader_iter = iter(self.reader)
        return self

    def __next__(self):
        reader_output = next(self.reader_iter)
        return self.alter_output(reader_output)

    @abstractmethod
    def alter_output(self, reader_output: Tuple[datetime, set]) -> Tuple[datetime, set]:
        ...


class BaseModifiedFeeder(Feeder):

    @BufferedGenerator.generator_method
    def data_gen(self):
        for reader_output in self.reader:
            yield self.alter_output(reader_output)
        return

    @abstractmethod
    def alter_output(self, reader_output: Tuple[datetime, set]) -> Tuple[datetime, set]:
        ...


class CopyFeeder(BaseModifiedFeeder):
    """
    Takes a copy of each object in the set and yields them. This is useful if you want to edit the
    object in the feeder later on but don't want to edit the original object
    """
    def alter_output(self, reader_output: Tuple[datetime, set]) -> Tuple[datetime, set]:
        time, set_of_items = reader_output
        copied_items = {copy.copy(item) for item in set_of_items}
        return time, copied_items


class DelayedFeeder(BaseModifiedFeeder):

    delay: timedelta = Property()

    def alter_output(self, reader_output: Tuple[datetime, set]) -> Tuple[datetime, set]:
        time, states = reader_output
        return time + self.delay, states
