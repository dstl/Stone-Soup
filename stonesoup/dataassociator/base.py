# -*- coding: utf-8 -*-
from ..base import Base
import itertools


class DataAssociator(Base):
    """Data Associator base class"""

    @staticmethod
    def isvalid(joint_hypotheses):
        number_hypotheses = len(joint_hypotheses)
        unique_hypotheses = len(set(
            [hyp.detection for hyp in joint_hypotheses]) - {None})
        number_null_hypotheses = sum(
            [hyp.detection is None for hyp in joint_hypotheses])

        if unique_hypotheses + number_null_hypotheses == number_hypotheses:
            return True
        else:
            return False

    @staticmethod
    def enumerate_joint_hypotheses(hypotheses):
        joint_hypotheses = list(itertools.product(*hypotheses.values()))
        joint_hypotheses = list(filter(
            DataAssociator.isvalid, joint_hypotheses))

        return joint_hypotheses
