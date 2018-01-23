# -*- coding: utf-8 -*-
from collections import defaultdict
import numpy as np

from .base import Hypothesiser
from ..types import Hypothesis


class LikelihoodRatioMatrixHypothesiser(Hypothesiser):
    def hypothesise(self, tracks, detections, clutter_param, gammaVal):
        hypotheses = defaultdict(list)
        for track in tracks:
            for detection in detections:
                state_vector, innov = self.update(track, detection)
                diff = np.linalg.lstsq(innov.covar, innov.state)
                mahabDist = np.sum(diff * diff)
                if mahabDist <= gammaVal:
                    hypothesis = Hypothesis(*state_vector)
                    hypothesis.detection = detection
                    hypothesis.ratio = track.det_prob/clutter_param*(1/((2*np.pi)**(len(innov.covar)/2)*np.abs(np.linalg.det(innov.covar))))*np.exp(-0.5*mahabDist)
                    hypotheses[track].append(hypothesis)
            # Missed detection ratio
            hypothesis = Hypothesis(*track.state_vector)
            hypothesis.ratio = 1 - track.det_prob
            hypotheses[track].append(hypothesis)
        return hypotheses
