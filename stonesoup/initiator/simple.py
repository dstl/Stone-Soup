import numpy as np
from scipy.stats import multivariate_normal

from .base import GaussianInitiator, ParticleInitiator
from ..base import Property
from ..dataassociator import DataAssociator
from ..deleter import Deleter
from ..models.base import NonLinearModel, ReversibleModel
from ..models.measurement import MeasurementModel
from ..types.hypothesis import SingleHypothesis
from ..types.numeric import Probability
from ..types.particle import Particle
from ..types.state import State, GaussianState
from ..types.track import Track
from ..types.update import GaussianStateUpdate, ParticleStateUpdate
from ..updater import Updater
from ..updater.kalman import ExtendedKalmanUpdater


class SinglePointInitiator(GaussianInitiator):
    """SinglePointInitiator class

    This uses an :class:`~.ExtendedKalmanUpdater` to carry out an update using
    provided :attr:`prior_state` for each unassociated detection.
    """

    prior_state: GaussianState = Property(doc="Prior state information")
    measurement_model: MeasurementModel = Property(doc="Measurement model")

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

        updater = ExtendedKalmanUpdater(self.measurement_model)

        tracks = set()
        for detection in unassociated_detections:
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

    def initiate(self, detections, **kwargs):
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
    initiator: SimpleMeasurementInitiator = Property(default=None,
                                                     doc="Initiator used to create tracks")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.holding_tracks = set()
        if self.initiator is None:
            self.initiator = SimpleMeasurementInitiator(self.prior_state, self.measurement_model)

    def initiate(self, detections, **kwargs):
        sure_tracks = set()
        if len(detections) == 0:
            return sure_tracks

        detections_list = list(detections)
        detections_set = set(detections)
        associated_detections = set()

        if not len(self.holding_tracks) == 0:
            associations = self.data_associator.associate(
                self.holding_tracks, detections, detections_list[0].timestamp)

            for track, hypothesis in associations.items():
                if hypothesis:
                    state_post = self.updater.update(hypothesis)
                    track.append(state_post)
                    if len(track) >= self.min_points:
                        new_track = Track(track[-1:])
                        sure_tracks.add(new_track)
                        self.holding_tracks.remove(track)
                    associated_detections.add(hypothesis.measurement)
                else:
                    track.append(hypothesis.prediction)

            self.holding_tracks -= \
                self.deleter.delete_tracks(self.holding_tracks)
        self.holding_tracks |= self.initiator.initiate(detections_set - associated_detections)

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

    def initiate(self, unassociated_detections, **kwargs):
        """Initiates tracks given unassociated measurements

        Parameters
        ----------
        unassociated_detections : list of :class:`~.Detection`
            A list of unassociated detections

        Returns
        -------
        : set of :class:`~.Track`
            A list of new tracks with a initial :class:`~.ParticleState`
        """
        tracks = self.initiator.initiate(unassociated_detections, **kwargs)
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
                timestamp=track.timestamp)

        return tracks
