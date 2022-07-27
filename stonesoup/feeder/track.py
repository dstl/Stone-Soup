# -*- coding: utf-8 -*-
import numpy as np

from stonesoup.types.detection import GaussianDetection
from stonesoup.feeder.base import DetectionFeeder
from stonesoup.models.measurement.linear import LinearGaussian
from ..buffered_generator import BufferedGenerator


class Tracks2GaussianDetectionFeeder(DetectionFeeder):
    '''
    Feeder consumes Track objects and outputs GaussianDetection objects.

    At each time step, the :attr:`Reader` feeds in a set of live tracks. The feeder takes the most
    recent state from each of those tracks, and turn them into a set of
    :class:`~.GaussianDetection` objects. Each detection is given a :class:`~.LinearGaussian`
    measurement model whose covariance is equal to the state covariance. The feeder assumes that
    the tracks are all live, that is each track has a state at the most recent time step.
    '''
    @BufferedGenerator.generator_method
    def data_gen(self):
        for time, tracks in self.reader:
            detections = []
            for track in tracks:
                dim = len(track.state.state_vector)
                detections.append(
                    GaussianDetection.from_state(
                        track.state,
                        measurement_model=LinearGaussian(dim, range(dim), np.asarray(track.covar)),
                        target_type=GaussianDetection)
                )
            print('Detections:', detections)
            yield time, detections
