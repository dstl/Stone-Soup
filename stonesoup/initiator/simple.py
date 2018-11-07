import numpy as np
from scipy.stats import multivariate_normal

from .base import Initiator, GaussianInitiator
from ..base import Property
from ..updater import KalmanUpdater
from ..models.measurement import MeasurementModel
from ..types.numeric import Probability
from ..types.particle import Particle
from ..types.state import GaussianState, ParticleState
from ..types.track import Track


class SinglePointInitiator(GaussianInitiator):
    """ SinglePointInitiator class"""

    prior_state = Property(GaussianState, doc="Prior state information")
    measurement_model = Property(MeasurementModel, doc="Measurement model")

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
            post_state_vec, post_state_covar, _ = \
                KalmanUpdater.update_lowlevel(self.prior_state.state_vector,
                                              self.prior_state.covar,
                                              self.measurement_model.matrix(),
                                              self.measurement_model.covar(),
                                              detection.state_vector)

            track_state = GaussianState(
                post_state_vec,
                post_state_covar,
                timestamp=detection.timestamp)
            track = Track([track_state])
            tracks.add(track)

        return tracks


class LinearMeasurementInitiator(Initiator):
    """Initiator that maps measurement space to state space

    This initiator utilises the :class:`~.MeasurementModel` matrix to convert
    :class:`~.Detection` state vector and model covariance into state space.
    This then replaces mapped values in the :attr:`prior_state` to form the
    initial :class:`~.GaussianState` of the :class:`~.Track`.
    """
    prior_state = Property(GaussianState, doc="Prior state information")
    measurement_model = Property(MeasurementModel, doc="Measurement model")

    def initiate(self, detections, **kwargs):
        tracks = set()

        model_matrix = self.measurement_model.matrix()
        model_covar = self.measurement_model.covar()

        # Zero out elements of prior state that will be replaced by measurement
        # (1 - matrix) to flip 0->1 and 1->0
        prior_state_vector = self.prior_state.state_vector * (
            1 - model_matrix.T@np.ones((model_matrix.shape[0], 1)))
        prior_covar = self.prior_state.covar * (
            1 - model_matrix.T@np.ones(model_covar.shape)@model_matrix)

        for detection in detections:
            tracks.add(Track([GaussianState(
                prior_state_vector + model_matrix.T@detection.state_vector,
                prior_covar + model_matrix.T@model_covar@model_matrix,
                timestamp=detection.timestamp)
            ]))
        return tracks


class GaussianParticleInitiator(Initiator):
    """Gaussian Particle Initiator class

    Utilising Gaussian Initiator, sample from the resultant track's state
    to generate a number of particles, overwriting with a
    :class:`~.ParticleState`.
    """

    initiator = Property(
        GaussianInitiator,
        doc="Gaussian Initiator which will be used to generate tracks.")
    number_particles = Property(
        float, default=200, doc="Number of particles for initial track state")

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
        weight = Probability(1/self.number_particles)
        for track in tracks:
            samples = multivariate_normal.rvs(track.state_vector.ravel(),
                                              track.covar,
                                              size=self.number_particles)
            particles = [
                Particle(sample.reshape(-1, 1), weight=weight)
                for sample in samples]
            track[-1] = ParticleState(particles,
                                      timestamp=track.timestamp)

        return tracks
