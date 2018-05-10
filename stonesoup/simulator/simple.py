# -*- coding: utf-8 -*-
import datetime

import numpy as np

from ..base import Property
from ..measurementmodel import MeasurementModel
from ..predictor import Predictor
from ..reader import GroundTruthReader
from ..types import (
    Detection, GaussianDetection, GaussianState, GroundTruthState,
    GroundTruthPath, Probability, State)
from .base import DetectionSimulator, GroundTruthSimulator


class SingleTargetGroundTruthSimulator(GroundTruthSimulator):
    """Target simulator that produces a single target"""
    predictor = Property(
        Predictor, doc="Predictor used as propagator for track.")
    initial_state = Property(
        State,
        default=State(np.array([[0], [0], [0], [0]])),
        doc="Initial state to use to generate ground truth")
    timestep = Property(
        datetime.timedelta,
        default=datetime.timedelta(seconds=1),
        doc="Time step between each state. Default one second.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.groundtruth_paths = set()

    def groundtruth_paths_gen(self):
        time = self.initial_state.timestamp or datetime.datetime.now()

        gttrack = GroundTruthPath([
            GroundTruthState(self.initial_state.state_vector, timestamp=time)])
        self.groundtruth_paths.add(gttrack)
        yield time, {gttrack}

        while True:
            time += self.timestep
            # Move track forward
            trans_state = self.predictor.predict(gttrack[-1], time)
            gttrack.append(GroundTruthState(
                trans_state.state_vector +
                np.sqrt(trans_state.covar) @
                np.random.randn(trans_state.ndim, 1),
                timestamp=time))

            yield time, {gttrack}


class MultiTargetGroundTruthSimulator(SingleTargetGroundTruthSimulator):
    """Target simulator that produces multiple targets.

    Targets are created and destroyed randomly, as defined by the biirth rate
    and death probability."""
    initial_state = Property(
        GaussianState,
        default=GaussianState(
            np.array([[0], [0], [0], [0]]),
            np.diag([10000, 10000, 10, 10])),
        doc="Initial state to use to generate states")
    birth_rate = Property(
        float, default=1.0, doc="Rate at which tracks are born. Expected "
        "number of occurrences (λ) in Poisson distribution. Default 1.0.")
    death_probability = Property(
        Probability, default=0.1,
        doc="Probability of track dying in each time step. Default 0.1.")

    def groundtruth_paths_gen(self):
        time = self.initial_state.timestamp or datetime.datetime.now()
        active_tracks = set()

        while True:
            time += self.timestep
            # Random drop tracks
            active_tracks -= set(
                gttrack
                for gttrack in active_tracks
                if np.random.rand() <= self.death_probability)

            # Move tracks forward
            for gttrack in active_tracks:
                trans_state = self.predictor.predict(gttrack[-1], time)
                gttrack.append(GroundTruthState(
                    time,
                    trans_state.state_vector +
                    np.sqrt(trans_state.covar) @
                    np.random.randn(trans_state.ndim, 1)))

            # Random create
            for _ in range(np.random.poisson(self.birth_rate)):
                gttrack = GroundTruthPath()
                gttrack.append(GroundTruthState(
                    self.initial_state.state_vector +
                    np.sqrt(self.initial_state.covar) @
                    np.random.randn(self.initial_state.ndim, 1),
                    timestamp=time))
                self.groundtruth_paths.add(gttrack)
                active_tracks.add(gttrack)

            yield time, active_tracks


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

    def detections_gen(self):
        # TODO: Measurement model
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        R = np.array([[10, 0], [0, 10]])
        meas_range = np.array([[-300, 300], [-300, 300]])
        probability_of_detect = 0.9
        clutter_rate = 2.0

        for time, tracks in self.groundtruth.groundtruth_paths_gen():
            detections = set()
            for track in tracks:
                if np.random.rand() < probability_of_detect:
                    detection = GaussianDetection(
                        H @ track[-1].state_vector +
                        np.sqrt(R) @ np.random.randn(R.shape[0], 1),
                        np.eye(*R.shape),
                        timestamp=track[-1].timestamp)
                    detection.clutter = False
                    detections.add(detection)
                    self.real_detections.add(detection)

            # generate clutter
            for _ in range(np.random.poisson(clutter_rate)):
                detection = GaussianDetection(
                    np.random.rand(R.shape[0], 1) *
                    np.diff(meas_range) + meas_range[:, :1],
                    np.eye(*R.shape),
                    timestamp=time)
                detections.add(detection)
                self.clutter_detections.add(detection)

            yield time, detections
