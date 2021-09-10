# -*- coding: utf-8 -*-
from abc import abstractmethod

from stonesoup.base import Base


class Stitcher(Base):
    @abstractmethod
    def stitch(self, *args, **kwargs):
        raise NotImplementedError
