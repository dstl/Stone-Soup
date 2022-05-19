# -*- coding: utf-8 -*-
from typing import Optional
import datetime
from typing import Sequence, Collection

import numpy as np
from ordered_set import OrderedSet

from ..base import Property
from ..models.measurement import MeasurementModel
from ..models.transition import TransitionModel
from ..reader import GroundTruthReader
from ..types.detection import TrueDetection, Clutter
from ..types.groundtruth import GroundTruthPath, GroundTruthState
from ..types.numeric import Probability
from ..types.state import GaussianState, State
from ..types.array import StateVector
from .base import DetectionSimulator, GroundTruthSimulator
from stonesoup.buffered_generator import BufferedGenerator


class SingleTargetGroundTruthSimulator(GroundTruthSimulator):
    """Target simulator that produces a single target"""
    transition_model: TransitionModel = Property(
        doc="Transition Model used as propagator for track.")
    initial_state: State = Property(doc="Initial state to use to generate ground truth")
    timestep: datetime.timedelta = Property(
        default=datetime.timedelta(seconds=1),
        doc="Time step between each state. Default one second.")
    number_steps: int = Property(default=100, doc="Number of time steps to run for")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.index = 0

    @BufferedGenerator.generator_method
    def groundtruth_paths_gen(self):
        time = self.initial_state.timestamp or datetime.datetime.now()

        gttrack = GroundTruthPath([
            GroundTruthState(self.initial_state.state_vector, timestamp=time,
                             metadata={"index": self.index})])
        yield time, {gttrack}

        for _ in range(self.number_steps - 1):
            time += self.timestep
            # Move track forward
            trans_state_vector = self.transition_model.function(
                gttrack[-1], noise=True, time_interval=self.timestep)
            gttrack.append(GroundTruthState(
                trans_state_vector, timestamp=time,
                metadata={"index": self.index}))
            yield time, {gttrack}


class SwitchOneTargetGroundTruthSimulator(SingleTargetGroundTruthSimulator):
    """Target simulator that produces a single target. This target switches
    between multiple transition models based on a markov matrix
    (:attr:`model_probs`)"""
    transition_models: Sequence[TransitionModel] = Property(
        doc="List of transition models to be used, ensure that they all have the same dimensions.")
    model_probs: np.ndarray = Property(doc="A matrix of probabilities.\
    The element in the ith row and the jth column is the probability of\
     switching from the ith transition model in :attr:`transition_models`\
     to the jth")
    seed: Optional[int] = Property(default=None, doc="Seed for random number generation."
                                                     " Default None")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.seed is not None:
            self.random_state = np.random.RandomState(self.seed)
        else:
            self.random_state = np.random.mtrand._rand

    @property
    def transition_model(self):
        self.index = self.random_state.choice(range(0, len(self.transition_models)),
                                              p=self.model_probs[self.index])
        return self.transition_models[self.index]


class MultiTargetGroundTruthSimulator(SingleTargetGroundTruthSimulator):
    """Target simulator that produces multiple targets.

    Targets are created and destroyed randomly, as defined by the birth rate
    and death probability."""
    transition_model: TransitionModel = Property(
        doc="Transition Model used as propagator for track.")
    initial_state: GaussianState = Property(doc="Initial state to use to generate states")
    birth_rate: float = Property(
        default=1.0, doc="Rate at which tracks are born. Expected number of occurrences (λ) in "
                         "Poisson distribution. Default 1.0.")
    death_probability: Probability = Property(
        default=0.1, doc="Probability of track dying in each time step. Default 0.1.")
    seed: Optional[int] = Property(default=None, doc="Seed for random number generation."
                                                     " Default None")
    preexisting_states: Collection[StateVector] = Property(
        default=list(), doc="State vectors at time 0 for "
                            "groundtruths which should exist at the start of simulation.")
    initial_number_targets: int = Property(
        default=0, doc="Initial number of targets to be "
                       "simulated. These simulated targets will be made in addition to those "
                       "defined by :attr:`preexisting_states`.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.seed is not None:
            self.random_state = np.random.RandomState(self.seed)
        else:
            self.random_state = np.random.mtrand._rand

    def _new_target(self, time, random_state, state_vector=None):
        vector = state_vector or \
            self.initial_state.state_vector + \
            self.initial_state.covar @ \
            random_state.randn(self.initial_state.ndim, 1)

        gttrack = GroundTruthPath()
        gttrack.append(GroundTruthState(
            state_vector=vector,
            timestamp=time,
            metadata={"index": self.index})
        )
        return gttrack

    @BufferedGenerator.generator_method
    def groundtruth_paths_gen(self, random_state=None):
        time = self.initial_state.timestamp or datetime.datetime.now()
        random_state = random_state if random_state is not None else self.random_state
        number_steps_remaining = self.number_steps

        if self.preexisting_states or self.initial_number_targets:
            # Use preexisting_states to make some groundtruth paths
            preexisting_paths = OrderedSet(
                self._new_target(time, random_state, state) for state in self.preexisting_states)

            # Simulate more groundtruth paths for the number of initial_simulated_states
            initial_simulated_paths = OrderedSet(
                self._new_target(time, random_state) for _ in range(self.initial_number_targets))

            # Union the two sets
            groundtruth_paths = preexisting_paths | initial_simulated_paths

            number_steps_remaining -= 1
            yield time, groundtruth_paths
            self.time += self.timestep

        else:
            groundtruth_paths = OrderedSet()

        for _ in range(number_steps_remaining):
            # Random drop tracks
            groundtruth_paths.difference_update(
                gttrack
                for gttrack in groundtruth_paths.copy()
                if random_state.rand() <= self.death_probability)

            # Move tracks forward
            for gttrack in groundtruth_paths:
                self.index = gttrack[-1].metadata.get("index")
                trans_state_vector = self.transition_model.function(
                    gttrack[-1], noise=True, time_interval=self.timestep)
                gttrack.append(GroundTruthState(
                    trans_state_vector, timestamp=time,
                    metadata={"index": self.index}))

            # Random create
            for _ in range(random_state.poisson(self.birth_rate)):
                self.index = 0
                gttrack = self._new_target(time, random_state)
                groundtruth_paths.add(gttrack)

            yield time, groundtruth_paths
            time += self.timestep


class SwitchMultiTargetGroundTruthSimulator(MultiTargetGroundTruthSimulator):
    """Functions identically to :class:`~.MultiTargetGroundTruthSimulator`,
    but has the transition model switching ability from
    :class:`.SwitchOneTargetGroundTruthSimulator`"""
    transition_models: Sequence[TransitionModel] = Property(
        doc="List of transition models to be used, ensure that they all have the same dimensions.")
    model_probs: np.ndarray = Property(doc="A matrix of probabilities.\
        The element in the ith row and the jth column is the probability of\
         switching from the ith transition model in :attr:`transition_models`\
         to the jth")
    seed: Optional[int] = Property(default=None, doc="Seed for random number generation."
                                                     " Default None")

    @property
    def transition_model(self, random_state=None):
        random_state = random_state if random_state is not None else self.random_state
        self.index = random_state.choice(range(0, len(self.transition_models)),
                                         p=self.model_probs[self.index])
        return self.transition_models[self.index]


class SimpleDetectionSimulator(DetectionSimulator):
    """A simple detection simulator.

    Parameters
    ----------
    groundtruth : GroundTruthReader
        Source of ground truth tracks used to generate detections for.
    measurement_model : MeasurementModel
        Measurement model used in generating detections.
    """
    groundtruth: GroundTruthReader = Property()
    measurement_model: MeasurementModel = Property()
    meas_range: np.ndarray = Property()
    detection_probability: Probability = Property(default=0.9)
    clutter_rate: float = Property(default=2.0)
    seed: Optional[int] = Property(default=None, doc="Seed for random number generation."
                                                     " Default None")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.real_detections = set()
        self.clutter_detections = set()
        self.index = 0
        if self.seed is not None:
            self.random_state = np.random.RandomState(self.seed)
        else:
            self.random_state = np.random.mtrand._rand

    @property
    def clutter_spatial_density(self):
        """returns the clutter spatial density of the measurement space - num
        clutter detections per unit volume per timestep"""
        return self.clutter_rate/np.prod(np.diff(self.meas_range))

    def __in_state_space(self, detection):
        """
        Checks if a measurement is in the state space
        """
        for dim in range(self.meas_range.ndim):
            if not self.meas_range[dim][0] <= detection.state_vector[dim] \
                                            <= self.meas_range[dim][-1]:
                return False
        return True

    @BufferedGenerator.generator_method
    def detections_gen(self, random_state=None):
        for time, tracks in self.groundtruth:
            self.real_detections.clear()
            self.clutter_detections.clear()
            random_state = random_state if random_state is not None else self.random_state
            for track in tracks:
                self.index = track[-1].metadata.get("index")
                if random_state.rand() < self.detection_probability:
                    detection = TrueDetection(
                        self.measurement_model.function(track[-1], noise=True),
                        timestamp=track[-1].timestamp,
                        groundtruth_path=track,
                        measurement_model=self.measurement_model)
                    detection.clutter = False
                    self.real_detections.add(detection)

            # generate clutter
            for _ in range(random_state.poisson(self.clutter_rate)):
                detection = Clutter(
                    random_state.rand(self.measurement_model.ndim_meas, 1) *
                    np.diff(self.meas_range) + self.meas_range[:, :1],
                    timestamp=time,
                    measurement_model=self.measurement_model)
                if self.__in_state_space(detection):
                    self.clutter_detections.add(detection)

            yield time, self.real_detections | self.clutter_detections


class SwitchDetectionSimulator(SimpleDetectionSimulator):

    """Functions identically as the :class:`SimpleDetectionSimulator`, but for
    ground truth paths formed using multiple transition models it allows the
    user to assign a detection probability to each transition models.
    For example, if you wanted a higher detection probability when the
    simulated object makes a turn"""

    detection_probabilities: Sequence[Probability] = Property(
        doc="List of probabilities that correspond to the detection probability of the simulated "
            "object while undergoing each transition model")

    @property
    def detection_probability(self):
        return self.detection_probabilities[self.index]


class DummyGroundTruthSimulator(GroundTruthSimulator):
    """A Dummy Ground Truth Simulator which allows simulations to be built
     where platform, rather than ground truth objects, motions are simulated.

     It returns an empty set at each time step.
    """

    times: Sequence[datetime.datetime] = Property(doc='list of times to return')

    @BufferedGenerator.generator_method
    def groundtruth_paths_gen(self):
        for time in self.times:
            yield time, set()
