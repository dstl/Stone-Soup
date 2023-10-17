import numpy as np
from scipy.stats import multivariate_normal
import datetime

from .base import GaussianInitiator, ParticleInitiator, Initiator
from ..base import Property
from ..dataassociator import DataAssociator
from ..deleter import Deleter
from ..models.base import LinearModel, ReversibleModel
from ..models.measurement import MeasurementModel
from ..types.array import StateVector, StateVectors
from ..types.hypothesis import SingleHypothesis
from ..types.mixture import GaussianMixture
from ..types.numeric import Probability
from ..types.particle import Particle
from ..types.state import State, GaussianState, ParticleState, TaggedWeightedGaussianState, \
    ASDGaussianState, EnsembleState
from ..types.track import Track
from ..types.update import GaussianStateUpdate, ParticleStateUpdate, Update, \
    GaussianMixtureUpdate, ASDGaussianStateUpdate, EnsembleStateUpdate
from ..updater import Updater
from ..updater.kalman import ExtendedKalmanUpdater


class SinglePointInitiator(GaussianInitiator):
    """SinglePointInitiator class

    This uses an :class:`~.ExtendedKalmanUpdater` to carry out an update using
    provided :attr:`prior_state` for each unassociated detection.
    """

    prior_state: GaussianState = Property(doc="Prior state information")
    measurement_model: MeasurementModel = Property(
        default=None,
        doc="Measurement model. Can be left as None if all detections have a "
            "valid measurement model.")

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
    It either takes the :class:`~.MeasurementModel` from the given detection
    or uses the :attr:`measurement_model`.

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
    measurement_model: MeasurementModel = Property(
        default=None,
        doc="Measurement model. Can be left as None if all detections have a "
            "valid measurement model.")
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
                if self.measurement_model is None:
                    raise ValueError("No measurement model specified")
                else:
                    measurement_model = self.measurement_model

            if isinstance(measurement_model, LinearModel):
                model_matrix = measurement_model.matrix()
                inv_model_matrix = np.linalg.pinv(model_matrix)
                state_vector = inv_model_matrix @ detection.state_vector
            else:
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
    deleter: Deleter = Property(doc="Deleter used to delete the track.")
    data_associator: DataAssociator = Property(
        doc="Association algorithm to pair predictions to detections.")
    updater: Updater = Property(
        doc="Updater used to update the track object to the new state.")
    measurement_model: MeasurementModel = Property(
        default=None,
        doc="Measurement model. Can be left as None if all detections have a "
            "valid measurement model.")
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

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        # Create prior particle state
        try:
            samples = multivariate_normal.rvs(self.initiator.prior_state.state_vector.ravel(),
                                              self.initiator.prior_state.covar,
                                              size=self.number_particles)
        except AttributeError:
            raise AttributeError("No prior state")
        particles = [
            Particle(sample.reshape(-1, 1), weight=self.weight)
            for sample in samples]

        self.prior_state = ParticleState(
            particles,
            fixed_covar=self.initiator.prior_state.covar if self.use_fixed_covar else None
        )

    @property
    def weight(self):
        return Probability(1 / self.number_particles)

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

        for track in tracks:
            samples = multivariate_normal.rvs(track.state_vector.ravel(),
                                              track.covar,
                                              size=self.number_particles)
            particles = [
                Particle(sample.reshape(-1, 1), weight=self.weight)
                for sample in samples]
            track[-1] = ParticleStateUpdate(
                None,
                track.hypothesis,
                particle_list=particles,
                fixed_covar=track.covar if self.use_fixed_covar else None,
                timestamp=track.timestamp)

        return tracks


class GaussianMixtureInitiator(GaussianInitiator):
    """Gaussian Mixture Initiator class

    Utilising Gaussian Initiator, applying the resultant track's state
    to generate a Tagged Weighted Gaussian State, overwriting with a
    :class:`~.GaussianMixture`.
    """

    initiator: GaussianInitiator = Property(
        doc="Gaussian Initiator which will be used to generate tracks.")

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        # Create prior particle state
        try:
            state = self.initiator.prior_state.state_vector
            covar = self.initiator.prior_state.covar

        except AttributeError:
            raise AttributeError("No prior state")

        self.prior_state = GaussianMixture([
            TaggedWeightedGaussianState(
                state_vector=state,
                covar=covar,
                weight=Probability(1),
                tag=[])])

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
            A list of new tracks with an initial :class:`~.GaussianMixture`
        """
        tracks = self.initiator.initiate(detections, timestamp, **kwargs)

        for track in tracks:
            mixture = [
                TaggedWeightedGaussianState(
                    state_vector=track.state_vector,
                    covar=track.covar,
                    weight=Probability(1),
                    timestamp=track.timestamp,
                    tag=[])]
            track[-1] = GaussianMixtureUpdate(
                hypothesis=track.hypothesis,
                components=mixture)

        return tracks


class ASDGaussianInitiator(GaussianInitiator):
    """ASD Gaussian State Initiator class

    Utilising Gaussian Initiator, sample from the resultant track's state
    to generate an ASD Gaussian State, overwriting with a
    :class:`~.ASDGaussianState`.
    """

    initiator: GaussianInitiator = Property(
        doc="Gaussian Initiator which will be used to generate tracks.")
    max_nstep: int = Property(
        default=0,
        doc="Decides when the state is pruned in a prediction step. If 0 then there is no pruning")

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        # Create prior particle state
        try:
            state = self.initiator.prior_state.state_vector
            covar = self.initiator.prior_state.covar

        except AttributeError:
            raise AttributeError("No prior state")

        self.prior_state = ASDGaussianState(multi_state_vector=state,
                                            timestamps=None,
                                            max_nstep=self.max_nstep,
                                            multi_covar=covar)

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
            A list of new tracks with an initial :class:`~.ASDGaussianState`
        """
        tracks = self.initiator.initiate(detections, timestamp, **kwargs)

        for track in tracks:
            state = track.state_vector
            covar = track.covar
            timestamp = track.timestamp

            track[-1] = ASDGaussianStateUpdate(
                multi_state_vector=state,
                timestamps=timestamp,
                max_nstep=self.max_nstep,
                multi_covar=covar,
                hypothesis=track.hypothesis)

        return tracks


class EnsembleInitiator(GaussianInitiator):
    """Ensemble State Initiator class

    Utilising Gaussian Initiator, sample from the resultant track's state
    to generate an Ensemble, overwriting with a
    :class:`~.EnsembleState`.
    """

    initiator: GaussianInitiator = Property(
        doc="Gaussian Initiator which will be used to generate tracks.")
    ensemble_size: int = Property(
        default=100,
        doc="Integer to determine the size of the Gaussian Ensemble State.")

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        # Create prior particle state
        try:
            state = self.initiator.prior_state

        except AttributeError:
            raise AttributeError("No prior state")

        self.prior_state = EnsembleState.from_gaussian_state(state, self.ensemble_size)

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
            A list of new tracks with an initial :class:`~.EnsembleState`
        """
        tracks = self.initiator.initiate(detections, timestamp, **kwargs)

        for track in tracks:
            gaussian_state = GaussianState(track.state_vector,
                                           track.covar,
                                           track.timestamp)

            ensemble = EnsembleState.from_gaussian_state(gaussian_state, self.ensemble_size)

            track[-1] = EnsembleStateUpdate(
                state_vector=ensemble.state_vector,
                hypothesis=track.hypothesis,
                timestamp=ensemble.timestamp)

        return tracks
