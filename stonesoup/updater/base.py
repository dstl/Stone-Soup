# -*- coding: utf-8 -*-
from abc import abstractmethod
from ..base import Base


class Updater(Base):
    """Updater base class"""
    @abstractmethod
    def update(self, *args, **kwargs):
        raise NotImplemented
