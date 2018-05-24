from .base import Initiator
from ..base import Property
from ..predictor import KalmanPredictor
from ..updater import KalmanUpdater
from ..models.measurementmodel import MeasurementModel
from ..types.track import Track
from ..types.state import GaussianState


class SinglePointInitiator(Initiator):
    """ SinglePointInitiator class"""

    prior_state = Property(GaussianState, doc="Prior state information")
    measurement_model = Property(MeasurementModel, doc="Measurement model")

    def initiate(self, unasssociated_detections, **kwargs):
        """Initiates tracks given unassociated measurements

        Parameters
        ----------
        unasssociated_detections : list of \
        :class:`stonesoup.types.detection.Detection`
            A list of unassociated detections

        Returns
        -------
        : :class:`sets.Set` of :class:`stonesoup.types.track.Track`
            A list of new tracks
        """

        tracks = set()
        for detection in unasssociated_detections:
            detection_state_vec = detection.state_vector
            meas_pred_state_vec, meas_pred_covar, cross_covar = \
                KalmanPredictor._predict_meas(self.prior_state.state_vector,
                                              self.prior_state.covar,
                                              self.measurement_model.matrix(),
                                              self.measurement_model.covar())
            post_state_vec, post_state_covar, kalman_gain = \
                KalmanUpdater._update(self.prior_state.state_vector,
                                      self.prior_state.covar,
                                      detection_state_vec,
                                      meas_pred_state_vec,
                                      meas_pred_covar,
                                      cross_covar)

            track_state = GaussianState(
                post_state_vec,
                post_state_covar,
                timestamp=detection.timestamp)
            track = Track()
            track.states.append(track_state)
            tracks.add(track)

        return tracks
