# -*- coding: utf-8 -*-
import numpy as np

from ..base import Property
from ..measurementmodel import MeasurementModel
from ..predictor import Predictor
from ..reader import GroundTruthReader
from ..types import (
    Detection, GroundTruthState, GroundTruthTrack, Probability, GaussianState)
from .base import DetectionSimulator, GroundTruthSimulator


class SimpleGroundTruthSimulator(GroundTruthSimulator):
    """A simple ground truth track simulator.
    """
    predictor = Property(
        Predictor, doc="Predictor used as propagator for track.")
    birth_rate = Property(
        float, default=1.0, doc="Rate at which tracks are born. Expected "
        "number of occurrences (λ) in Poisson distribution. Default 1.0.")
    death_probability = Property(
        Probability, default=0.1,
        doc="Probability of track dying in each time step. Default 0.1.")
    initial_state = Property(
        GaussianState,
        default=GaussianState(
            np.array([[0], [0], [0], [0]]),
            np.diag([10000, 10000, 10, 10])),
        doc="Initial state to use to generate states")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tracks = set()

    def get_tracks(self):
        active_tracks = set()

        while True:
            # Random drop tracks
            active_tracks -= set(
                gttrack
                for gttrack in active_tracks
                if np.random.rand() <= self.death_probability)

            # Move tracks forward
            for gttrack in active_tracks:
                trans_state = self.predictor.predict(gttrack[-1])
                gttrack.append(GroundTruthState(
                    trans_state.state_vector +
                    np.sqrt(trans_state.covar) @
                    np.random.randn(trans_state.ndim, 1)))

            # Random create
            for _ in range(np.random.poisson(self.birth_rate)):
                gttrack = GroundTruthTrack()
                gttrack.append(GroundTruthState(
                    self.initial_state.state_vector +
                    np.sqrt(self.initial_state.covar) @
                    np.random.randn(self.initial_state.ndim, 1)))
                self.tracks.add(gttrack)
                active_tracks.add(gttrack)

            yield active_tracks


class SimpleDetectionSimulator(DetectionSimulator):
    """A simple detection simulator.

    Parameters
    ----------
    groundtruth : GroundTruthReader
        Source of ground truth tracks used to generate detections for.
    measurement_model : MeasurementModel
        Measurement model used in generating detections.
    """
    groundtruth = Property(GroundTruthReader)
    measurement_model = Property(MeasurementModel)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.real_detections = set()
        self.clutter_detections = set()

    @property
    def detections(self):
        return self.real_detections | self.clutter_detections

    def get_detections(self):
        # TODO: Measurement model
        H = np.array([[1, 0], [0, 1]])
        R = np.array([[0.001, 0], [0, 0.001]])
        meas_range = np.array([[-300, 300], [-10, 10]])
        probability_of_detect = 0.9
        clutter_rate = 2.0

        for tracks in self.groundtruth.get_tracks():
            detections = set()
            for track in tracks:
                if np.random.rand() > probability_of_detect:
                    track.detections.append(None)
                else:
                    detection = Detection(
                        H @ track.groundtruth[-1].state +
                        np.sqrt(R) @ np.random.randn(R.shape[0], 1),
                        np.eye(*R.shape))
                    detection.source = self
                    detection.clutter = False
                    track.detections.append(detection)

                    detections.add(detection)
                    self.real_detections.add(detection)

            # generate clutter
            for _ in range(np.random.poisson(clutter_rate)):
                detection = Detection(
                    np.random.rand(R.shape[0], 1) *
                    np.diff(meas_range) + meas_range[:, :1],
                    np.eye(*R.shape))
                detections.add(detection)
                self.clutter_detections.add(detection)

            yield detections
