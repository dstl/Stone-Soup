# -*- coding: utf-8 -*-
import pytest

from ...types import DistanceHypothesis, GaussianStatePrediction,\
    GaussianMeasurementPrediction


@pytest.fixture()
def hypothesiser():
    class TestGaussianHypothesiser:
        def hypothesise(self, track, detections, timestamp):
            hypotheses = list()
            for detection in detections:
                prediction = GaussianStatePrediction(track.state_vector + 1,
                                                     track.covar * 2,
                                                     detection.timestamp)
                measurement_prediction =\
                    GaussianMeasurementPrediction(prediction.state_vector,
                                                  prediction.covar,
                                                  prediction.timestamp)
                distance = abs(track.state_vector - detection.state_vector)

                hypotheses.append(DistanceHypothesis(
                    prediction, detection, distance, measurement_prediction))

            prediction = GaussianStatePrediction(track.state_vector + 1,
                                                 track.covar * 2, timestamp)
            hypotheses.append(DistanceHypothesis(prediction, None, 10))
            return hypotheses
    return TestGaussianHypothesiser()
