# -*- coding: utf-8 -*-
import datetime

import numpy as np

from ..base import Property
from ..models import MeasurementModel
from ..models import TransitionModel
from ..reader import GroundTruthReader
from ..types import (Detection, GaussianState, GroundTruthState,
                     GroundTruthPath, Probability, State)
from .base import DetectionSimulator, GroundTruthSimulator


class SingleTargetGroundTruthSimulator(GroundTruthSimulator):
    """Target simulator that produces a single target"""
    transition_model = Property(
        TransitionModel, doc="Transition Model used as propagator for track.")
    initial_state = Property(
        State,
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
            trans_state_vector = self.transition_model.function(
                gttrack[-1].state_vector,
                time_interval=self.timestep)
            gttrack.append(GroundTruthState(
                trans_state_vector, timestamp=time))

            yield time, {gttrack}


class MultiTargetGroundTruthSimulator(SingleTargetGroundTruthSimulator):
    """Target simulator that produces multiple targets.

    Targets are created and destroyed randomly, as defined by the biirth rate
    and death probability."""
    initial_state = Property(
        GaussianState,
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
                trans_state_vector = self.transition_model.function(
                    gttrack[-1].state_vector,
                    time_interval=self.timestep)
                gttrack.append(GroundTruthState(
                    trans_state_vector, timestamp=time))

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
    meas_range = Property(np.ndarray)
    probability_of_detect = Property(Probability, default=0.9)
    clutter_rate = Property(float, default=2.0)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.real_detections = set()
        self.clutter_detections = set()

    @property
    def detections(self):
        return self.real_detections | self.clutter_detections

    def detections_gen(self):
        H = self.measurement_model.matrix()

        for time, tracks in self.groundtruth.groundtruth_paths_gen():
            detections = set()
            for track in tracks:
                if np.random.rand() < self.probability_of_detect:
                    detection = Detection(
                        H @ track[-1].state_vector +
                        self.measurement_model.rvs(),
                        timestamp=track[-1].timestamp)
                    detection.clutter = False
                    detections.add(detection)
                    self.real_detections.add(detection)

            # generate clutter
            for _ in range(np.random.poisson(self.clutter_rate)):
                detection = Detection(
                    np.random.rand(H.shape[0], 1) *
                    np.diff(self.meas_range) + self.meas_range[:, :1],
                    timestamp=time)
                detections.add(detection)
                self.clutter_detections.add(detection)

            yield time, detections
