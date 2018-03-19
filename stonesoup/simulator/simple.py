# -*- coding: utf-8 -*-
import numpy as np

from .base import DetectionSimulator, GroundTruthSimulator
from ..base import Property
from ..measurementmodel import MeasurementModel
from ..reader import GroundTruthReader
from ..transitionmodel import TransitionModel
from ..types import Detection, GroundTruth, Probability, StateVector, Track


class SimpleGroundTruthSimulator(GroundTruthSimulator):
    """A simple ground truth track simulator.

    Parameters
    ----------
    transition_model : TransitionModel
        TransitionModel used for propagating tracks.
    birth_rate : float, optional
        Rate at which tracks are born. Expected number of occurrences (λ) in
        Poisson distribution. Default 1.0
    death_probability : Probability, optional
        Probability of track dying in each time step. Default 0.1
    initial_state : StateVector, optional
        Initial starting state for born tracks.
    """
    transition_model = Property(TransitionModel)
    birth_rate = Property(float, default=1.0)
    death_probability = Property(Probability, default=0.1)
    initial_state = Property(StateVector, default=StateVector(
        np.array([[0], [0]]),
        np.array([[10000, 0], [0, 10]])))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tracks = set()

    def get_tracks(self):
        active_tracks = set()
        # TODO: Use transition model
        F = np.array([[1, 1], [0, 1]])
        Q = np.array([[1/3, 1/2], [1/2, 1]]) * 0.0002

        while True:
            # Random drop tracks
            active_tracks -= set(
                track
                for track in active_tracks
                if np.random.rand() <= self.death_probability)

            # Move tracks forward
            for track in active_tracks:
                track.groundtruth.append(GroundTruth(
                    F @ track.groundtruth[-1].state +
                    np.sqrt(Q) @ np.random.randn(Q.shape[0], 1)))

            # Random create
            for _ in range(np.random.poisson(self.birth_rate)):
                track = Track()
                track.groundtruth = [GroundTruth(
                    self.initial_state.state +
                    np.sqrt(self.initial_state.covar) @
                    np.random.randn(self.initial_state.covar.shape[0], 1))]
                self.tracks.add(track)
                active_tracks.add(track)

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
