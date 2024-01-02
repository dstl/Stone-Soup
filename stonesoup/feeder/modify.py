import copy
from abc import abstractmethod
from datetime import datetime, timedelta
from typing import Tuple

from .base import Feeder
from ..base import Property
from ..buffered_generator import BufferedGenerator


class BaseModifiedFeeder(Feeder):
    """ This class takes an object from a reader and alters it before releasing it."""

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
    """Changes the time value."""

    delay: timedelta = Property()

    def alter_output(self, reader_output: Tuple[datetime, set]) -> Tuple[datetime, set]:
        time, states = reader_output
        return time + self.delay, states
