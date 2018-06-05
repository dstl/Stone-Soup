# -*- coding: utf-8 -*-

from abc import abstractclassmethod
from collections import UserDict

from .base import Type
from ..types import DistanceHypothesis


class JointHypothesis(Type, UserDict):
    """Joint Hypothesis base type

    Parameters
    ----------

    """

    def __new__(cls, hypotheses):
        if all(isinstance(hypothesis, DistanceHypothesis)
               for hypothesis in hypotheses.values()):
            return super().__new__(DistanceJointHypothesis)

    @abstractclassmethod
    def __lt__(self, other):
        raise NotImplemented

    @abstractclassmethod
    def __le__(self, other):
        raise NotImplemented

    @abstractclassmethod
    def __eq__(self, other):
        raise NotImplemented

    @abstractclassmethod
    def __gt__(self, other):
        raise NotImplemented

    @abstractclassmethod
    def __ge__(self, other):
        raise NotImplemented


class DistanceJointHypothesis(JointHypothesis):
    """Distance scored hypothesis subclass.

        Notes
        -----
        As smaller distance is 'better', comparison logic is reversed
        i.e. smaller distance is a greater likelihood.

        Parameters
        ----------

        """

    def __init__(self, hypotheses):
        super().__init__(hypotheses)
        self.distance = sum(
            hypothesis.distance for hypothesis in self.data.values())

    def __lt__(self, other):
        return self.distance > other.distance

    def __le__(self, other):
        return self.distance >= other.distance

    def __eq__(self, other):
        return self.distance == other.distance

    def __gt__(self, other):
        return self.distance < other.distance

    def __ge__(self, other):
        return self.distance <= other.distance
