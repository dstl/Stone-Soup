# -*- coding: utf-8 -*-
from ..base import Base
from ..types import JointHypothesis
import itertools


class DataAssociator(Base):
    """Data Associator base class"""

    @staticmethod
    def isvalid(joint_hypothesis):
        number_hypotheses = len(joint_hypothesis)
        unique_hypotheses = len(
            {hyp.detection for hyp in joint_hypothesis} - {None})
        number_null_hypotheses = sum(
            hyp.detection is None for hyp in joint_hypothesis)

        if unique_hypotheses + number_null_hypotheses == number_hypotheses:
            return True
        else:
            return False

    @classmethod
    def enumerate_joint_hypotheses(cls, hypotheses):
        joint_hypotheses = [JointHypothesis({track: hypothesis for track, hypothesis in
                            zip(hypotheses, joint_hypothesis)}) for
                            joint_hypothesis in
                            itertools.product(*hypotheses.values()) if
                            cls.isvalid(joint_hypothesis)]

        return joint_hypotheses
