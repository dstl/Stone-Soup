from scipy.stats import multivariate_normal

from .base import Initiator
from ..base import Property
from ..updater import KalmanUpdater
from ..models.measurement import MeasurementModel
from ..types.track import Track
from ..types.state import GaussianState
from ..types.particle import Particle, ParticleState


class SinglePointInitiator(Initiator):
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
            track = Track()
            track.states.append(track_state)
            tracks.add(track)

        return tracks


class SinglePointParticleInitiator(SinglePointInitiator):
    """SinglePointParticleInitiator class

    Utilising the SinglePointInitiator, sample from the resultant track's state
    to generate a number of particles, overwriting with a
    :class:`~.ParticleState`.
    """

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
        tracks = super().initiate(unassociated_detections, **kwargs)
        for track in tracks:
            samples = multivariate_normal.rvs(track.state_vector.ravel(),
                                              track.covar,
                                              size=self.number_particles)
            particles = [
                Particle(sample.reshape(-1, 1), weight=1/self.number_particles)
                for sample in samples]
            track.states[-1] = ParticleState(particles,
                                             timestamp=track.timestamp)

        return tracks
