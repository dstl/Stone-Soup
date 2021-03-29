from typing import Sequence

import numpy as np
from scipy.stats import multivariate_normal

from ..types.detection import MissedDetection
from ..updater.composite import CompositeUpdater
from .base import GaussianInitiator, ParticleInitiator, Initiator
from ..base import Property
from ..dataassociator import DataAssociator
from ..deleter import Deleter
from ..models.base import NonLinearModel, ReversibleModel
from ..models.measurement import MeasurementModel
from ..types.hypothesis import SingleHypothesis, CompositeProbabilityHypothesis, \
    CompositeHypothesis
from ..types.numeric import Probability
from ..types.particle import Particle
from ..types.state import State, GaussianState, CompositeState
from ..types.track import Track
from ..types.update import GaussianStateUpdate, ParticleStateUpdate, Update, CompositeUpdate,\
    StateUpdate
from ..updater import Updater
from ..updater.kalman import ExtendedKalmanUpdater


class SinglePointInitiator(GaussianInitiator):
    """SinglePointInitiator class

    This uses an :class:`~.ExtendedKalmanUpdater` to carry out an update using
    provided :attr:`prior_state` for each unassociated detection.
    """

    prior_state: GaussianState = Property(doc="Prior state information")
    measurement_model: MeasurementModel = Property(doc="Measurement model")

    def initiate(self, detections, timestamp, **kwargs):
        """Initiates tracks given unassociated measurements

        Parameters
        ----------
        detections : set of :class:`~.Detection`
            A list of unassociated detections
        timestamp: datetime.datetime
            Current timestamp

        Returns
        -------
        : set of :class:`~.Track`
            A list of new tracks with an initial :class:`~.GaussianState`
        """

        updater = ExtendedKalmanUpdater(self.measurement_model)

        tracks = set()
        for detection in detections:
            measurement_prediction = updater.predict_measurement(
                self.prior_state, detection.measurement_model)
            track_state = updater.update(SingleHypothesis(
                self.prior_state, detection, measurement_prediction))
            track = Track([track_state])
            tracks.add(track)

        return tracks


class SimpleMeasurementInitiator(GaussianInitiator):
    """Initiator that maps measurement space to state space

    Works for both linear and non-linear co-ordinate input

    This initiator utilises the :class:`~.MeasurementModel` matrix to convert
    :class:`~.Detection` state vector and model covariance into state space.

    Utilises the ReversibleModel inverse function to convert
    non-linear spherical co-ordinates into Cartesian x/y co-ordinates
    for use in predictions and mapping.

    This then replaces mapped values in the :attr:`prior_state` to form the
    initial :class:`~.GaussianState` of the :class:`~.Track`.

    The diagonal loading value is used to try to ensure that the estimated
    covariance matrix is positive definite, especially for subsequent Cholesky
    decompositions.
    """
    prior_state: GaussianState = Property(doc="Prior state information")
    measurement_model: MeasurementModel = Property(doc="Measurement model")
    skip_non_reversible: bool = Property(default=False)
    diag_load: float = Property(default=0.0, doc="Positive float value for diagonal loading")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.diag_load < 0:
            raise ValueError(
                "diag_load value can't be less than 0.0")

    def initiate(self, detections, timestamp, **kwargs):
        tracks = set()

        for detection in detections:
            if detection.measurement_model is not None:
                measurement_model = detection.measurement_model
            else:
                measurement_model = self.measurement_model

            if isinstance(measurement_model, NonLinearModel):
                if isinstance(measurement_model, ReversibleModel):
                    try:
                        state_vector = measurement_model.inverse_function(
                            detection)
                    except NotImplementedError:
                        if not self.skip_non_reversible:
                            raise
                        else:
                            continue
                    model_matrix = measurement_model.jacobian(State(
                        state_vector))
                    inv_model_matrix = np.linalg.pinv(model_matrix)
                elif self.skip_non_reversible:
                    continue
                else:
                    raise Exception("Invalid measurement model used.\
                                    Must be instance of linear or reversible.")
            else:
                model_matrix = measurement_model.matrix()
                inv_model_matrix = np.linalg.pinv(model_matrix)
                state_vector = inv_model_matrix @ detection.state_vector

            model_covar = measurement_model.covar()

            prior_state_vector = self.prior_state.state_vector.copy()
            prior_covar = self.prior_state.covar.copy()

            mapped_dimensions = measurement_model.mapping

            prior_state_vector[mapped_dimensions, :] = 0
            prior_covar[mapped_dimensions, :] = 0
            C0 = inv_model_matrix @ model_covar @ inv_model_matrix.T
            C0 = C0 + prior_covar + np.diag(np.array([self.diag_load] * C0.shape[0]))
            tracks.add(Track([GaussianStateUpdate(
                prior_state_vector + state_vector,
                C0,
                SingleHypothesis(None, detection),
                timestamp=detection.timestamp)
            ]))
        return tracks


class MultiMeasurementInitiator(GaussianInitiator):
    """Multi-measurement initiator.

    Utilises features of the tracker to initiate and hold tracks
    temporarily within the initiator itself, releasing them to the
    tracker once there are multiple detections associated with them
    enough to determine that they are 'sure' tracks.

    Utilises simple initiator to initiate tracks to hold ->
    prevents code duplication.

    Solves issue of short-lived single detection tracks being
    initiated only to then be removed shortly after.
    Does cause slight delay in initiation to tracker."""

    prior_state: GaussianState = Property(doc="Prior state information")
    measurement_model: MeasurementModel = Property(doc="Measurement model")
    deleter: Deleter = Property(doc="Deleter used to delete the track.")
    data_associator: DataAssociator = Property(
        doc="Association algorithm to pair predictions to detections.")
    updater: Updater = Property(
        doc="Updater used to update the track object to the new state.")
    min_points: int = Property(
        default=2, doc="Minimum number of track points required to confirm a track.")
    updates_only: bool = Property(
        default=True, doc="Whether :attr:`min_points` only counts :class:`~.Update` states.")
    initiator: Initiator = Property(
        default=None,
        doc="Initiator used to create tracks. If None, a :class:`SimpleMeasurementInitiator` will "
            "be created using :attr:`prior_state` and :attr:`measurement_model`. Otherwise, these "
            "attributes are ignored.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.holding_tracks = set()
        if self.initiator is None:
            self.initiator = SimpleMeasurementInitiator(self.prior_state, self.measurement_model)

    def initiate(self, detections, timestamp, **kwargs):
        sure_tracks = set()

        associated_detections = set()

        if self.holding_tracks:
            associations = self.data_associator.associate(
                self.holding_tracks, detections, timestamp)

            for track, hypothesis in associations.items():
                if hypothesis:
                    state_post = self.updater.update(hypothesis)
                    track.append(state_post)
                    associated_detections.add(hypothesis.measurement)
                else:
                    track.append(hypothesis.prediction)

                if sum(1 for state in track if not self.updates_only or isinstance(state, Update))\
                        >= self.min_points:
                    sure_tracks.add(track)
                    self.holding_tracks.remove(track)

            self.holding_tracks -= self.deleter.delete_tracks(self.holding_tracks)

        self.holding_tracks |= self.initiator.initiate(
            detections - associated_detections, timestamp)

        return sure_tracks


class GaussianParticleInitiator(ParticleInitiator):
    """Gaussian Particle Initiator class

    Utilising Gaussian Initiator, sample from the resultant track's state
    to generate a number of particles, overwriting with a
    :class:`~.ParticleState`.
    """

    initiator: GaussianInitiator = Property(
        doc="Gaussian Initiator which will be used to generate tracks.")
    number_particles: int = Property(
        default=200, doc="Number of particles for initial track state")
    use_fixed_covar: bool = Property(
        default=False,
        doc="If `True`, the Gaussian state covariance is used for the "
            ":class:`~.ParticleState` as a fixed covariance. Default `False`.")

    def initiate(self, detections, timestamp, **kwargs):
        """Initiates tracks given unassociated measurements

        Parameters
        ----------
        detections : set of :class:`~.Detection`
            A list of unassociated detections
        timestamp: datetime.datetime
            Current timestamp

        Returns
        -------
        : set of :class:`~.Track`
            A list of new tracks with a initial :class:`~.ParticleState`
        """
        tracks = self.initiator.initiate(detections, timestamp, **kwargs)
        weight = Probability(1 / self.number_particles)
        for track in tracks:
            samples = multivariate_normal.rvs(track.state_vector.ravel(),
                                              track.covar,
                                              size=self.number_particles)
            particles = [
                Particle(sample.reshape(-1, 1), weight=weight)
                for sample in samples]
            track[-1] = ParticleStateUpdate(
                particles,
                track.hypothesis,
                fixed_covar=track.covar if self.use_fixed_covar else None,
                timestamp=track.timestamp)

        return tracks


class SimpleObservationInitiator(Initiator):
    prior_state: GaussianState = Property(doc="Prior state information")
    measurement_model: MeasurementModel = Property(default=None, doc="Measurement model (should "
                                                                     "be observation model)")

    def initiate(self, detections, **kwargs):
        tracks = set()

        for detection in detections:
            if detection.measurement_model is not None:
                measurement_model = detection.measurement_model
            else:
                measurement_model = self.measurement_model

            state_vector = measurement_model.inverse_function(detection)

            tracks.add(Track([
                StateUpdate(state_vector,
                            SingleHypothesis(None, detection),
                            timestamp=detection.timestamp)
            ]))
        return tracks


class SimpleCompositeInitiator(Initiator):
    prior_state: GaussianState = Property(doc="Prior state information")
    measurement_model: MeasurementModel = Property(doc="Measurement model")
    updater: CompositeUpdater = Property()

    def initiate(self, unassociated_detections, **kwargs):
        """Initiates tracks given unassociated measurements

        Parameters
        ----------
        unassociated_detections : list of \
        :class:`stonesoup.types.detection.Detection`
            A list of unassociated detections

        Returns
        -------
        : :class:`sets.Set` of :class:`stonesoup.types.track.Track`
            A list of new tracks with an initial :class:`~.GaussianState`
        """

        tracks = set()
        for detection in unassociated_detections:

            hypothesis = CompositeHypothesis(prediction=self.prior_state, measurement=detection)
            for i, sub_state in enumerate(self.prior_state.inner_states):

                try:
                    # Check if detection has sub-detection for this state index
                    sub_detection_index = detection.mapping.index(i)
                except IndexError:
                    hypothesis.append(SingleHypothesis(sub_state, MissedDetection(
                        timestamp=detection.timestamp)))
                else:
                    sub_detection = detection[sub_detection_index]
                    hypothesis.append(SingleHypothesis(sub_state, sub_detection))

            track_state = self.updater.update(hypothesis)
            track = Track([track_state])
            tracks.add(track)

        return tracks


class CompositeUpdateInitiator(Initiator):
    initiators: Sequence[Initiator] = Property()
    prior_state: CompositeState = Property(doc="Prior state information")

    def initiate(self, detections, **kwargs):
        tracks = set()

        for detection in detections:

            mapping = detection.mapping

            hypotheses = list()
            states = list()

            for i, initiator in enumerate(self.initiators):

                try:
                    # Check if detection has sub-detection for this state index
                    detection_index = mapping.index(i)
                except IndexError:
                    prior = self.prior_state[i]
                    states.append(prior)

                    # Add missed detection hypothesis to composite hypothesis
                    hypotheses.append(
                        SingleHypothesis(None, MissedDetection(timestamp=detection.timestamp)))
                else:
                    # Get sub-detection and initiate a (sub)track with it
                    sub_detection = detection[detection_index]
                    tracks = initiator.initiate({sub_detection})
                    track = tracks.pop()  # Set of 1 track
                    update = track[0]  # Get first state of track
                    states.append(update)

                    # Add detection hypothesis to composite hypothesis
                    hypotheses.append(SingleHypothesis(None, sub_detection))

            hypothesis = CompositeHypothesis(prediction=None,
                                             hypotheses=hypotheses,
                                             measurement=detection)
            composite_update = CompositeUpdate(inner_states=states,
                                               hypothesis=hypothesis)

            tracks.add(Track([composite_update]))
        return tracks
