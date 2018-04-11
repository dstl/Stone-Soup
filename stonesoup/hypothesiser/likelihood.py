# -*- coding: utf-8 -*-
from collections import defaultdict

import numpy as np

from .base import Hypothesiser
from ..types import Hypothesis


# TODO: Complete documentation
class LikelihoodRatioMatrixHypothesiser(Hypothesiser):
    """Likelihood Ratio Matrix Hypothesiser"""
    def hypothesise(self, tracks, detections, clutter_param, gammaVal):
        hypotheses = defaultdict(list)
        for track in tracks:
            for detection in detections:
                state_vector, innov = self.updater.update(track, detection)
                diff = np.linalg.lstsq(innov.covar, innov.state)[0]
                mahabDist = np.sum(diff * diff)
                if mahabDist <= gammaVal:
                    hypothesis = Hypothesis(
                        state_vector.state, state_vector.covar)
                    hypothesis.detection = detection
                    hypothesis.ratio = track.det_prob/clutter_param*(1/((2*np.pi)**(len(innov.covar)/2)*np.abs(np.linalg.det(innov.covar))))*np.exp(-0.5*mahabDist)
                    hypotheses[track].append(hypothesis)
            # Missed detection ratio
            hypothesis = Hypothesis(track.state, track.covar)
            hypothesis.ratio = 1 - track.det_prob
            hypotheses[track].append(hypothesis)
        return hypotheses
