import numpy as np
from abc import abstractmethod
from ..base import Base, Property
from ..dataassociator.neighbour import NearestNeighbour
from ..deleter.error import CovarianceBasedDeleter
from ..hypothesiser.distance import DistanceHypothesiser
from ..initiator.simple import OnePointInitiator, SinglePointInitiator
from ..measures import Euclidean
from ..predictor.kalman import ExtendedKalmanPredictor
from .simple import Tracker
from .simple import SingleTargetTracker
from ..updater.kalman import ExtendedKalmanUpdater
from ..types.state import StateVector
from ..models.transition import TransitionModel
from ..dataassociator import DataAssociator
from ..deleter import Deleter
from ..reader import DetectionReader
from ..initiator import Initiator
from ..updater import Updater
from ..predictor import Predictor


class PreBuiltTracker(Base):
    r"""PreBuiltTracker Base Class

    Todo

    """

    tracker: Tracker = Property(default=None, doc="Todo")

    @classmethod
    def get_tracker(cls, *args, **kwargs):
        tracker_builder = cls(*args, **kwargs)
        return tracker_builder.tracker


class PreBuiltSingleTargetTrackerNoClutter(PreBuiltTracker):

    # Require Components
    detector: DetectionReader = Property(doc="Detector used to generate detection objects.")

    # Optional Components
    initiator: Initiator = Property(default=None, doc="Initiator used to initialise the track.")
    deleter: Deleter = Property(default=None, doc="Deleter used to delete the track")
    data_associator: DataAssociator = Property(default=None, doc="Association algorithm to pair predictions to detections")
    updater: Updater = Property(default=None, doc="Updater used to update the track object to the new state.")
    predictor: Predictor = Property(default=None, doc="Predict tracks to detection times")

    # Required if none of the optional components are supplied
    ground_truth_prior: StateVector = Property(default=None, doc="Prior state information")
    target_transition_model: TransitionModel = Property(default=None, doc="The transition model to be used.")

    # Other Parameters that could be changed
    covar_deletion_threshold: float = Property(default=np.inf, doc="Covariance matrix trace threshold")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.initiator is None:
            if self.ground_truth_prior is not None:
                self.initiator = OnePointInitiator(self.ground_truth_prior, None)
            else:
                raise ValueError("If no initiator is provided then a ground truth prior must be provided")

        if self.deleter is None:
            self.deleter = CovarianceBasedDeleter(self.covar_deletion_threshold)

        if self.predictor is None:
            if self.target_transition_model is not None:
                self.predictor = ExtendedKalmanPredictor(self.target_transition_model)
            else:
                raise ValueError("If no predictor is provided then a transition model must be provided")

        if self.updater is None:
            self.updater = ExtendedKalmanUpdater(measurement_model=None)

        if self.data_associator is None:
            self.data_associator = NearestNeighbour(DistanceHypothesiser(
                    self.predictor, self.updater, Euclidean(), missed_distance=np.inf))

        self.tracker = SingleTargetTracker(
            self.initiator, self.deleter, self.detector, self.data_associator, self.updater)

