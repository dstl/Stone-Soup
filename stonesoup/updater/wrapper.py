from . import Updater
from ..base import Property
from ..types.detection import GaussianDetection


class DetectionAndTrackSwitchingUpdater(Updater):

    detection_updater: Updater = Property()
    track_updater: Updater = Property()

    def predict_measurement(self, state_prediction, measurement_model=None, **kwargs):
        if measurement_model.ndim == state_prediction.ndim:
            return self.track_updater.predict_measurement(
                state_prediction, measurement_model, **kwargs)
        else:
            return self.detection_updater.predict_measurement(
               state_prediction, measurement_model, **kwargs)

    def update(self, hypothesis, **kwargs):
        if isinstance(hypothesis.measurement, GaussianDetection):
            return self.track_updater.update(hypothesis, **kwargs)
        else:
            return self.detection_updater.update(hypothesis, **kwargs)
