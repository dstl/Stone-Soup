import numpy as np

from ...base import Base, Property
from ...models.base import LinearModel, ReversibleModel
from ...models.transition import TransitionModel
from ...types.state import GaussianState, State
from ...updater import Updater
from ..functions import predict_state_to_two_state, nearestPD, isPD
from ...types.hypothesis import SingleProbabilityHypothesis
from ...types.track import Track
from ...types.numeric import Probability

from ..types.state import TwoStateGaussianState


class TwoStateInitiator(Base):

    def __init__(self, *args, **kwargs):
        super(TwoStateInitiator, self).__init__(*args, **kwargs)
        self._max_track_id = 0

    prior: GaussianState = Property(doc='The prior used to initiate fused tracks')
    transition_model: TransitionModel = Property(doc='The transition model')
    updater: Updater = Property(doc='Updater used to update fused tracks')

    def initiate(self, detections, start_time, end_time, **kwargs):
        init_mean = self.prior.mean
        init_cov = self.prior.covar
        init_mean, init_cov = predict_state_to_two_state(init_mean, init_cov,
                                                         self.transition_model,
                                                         end_time - start_time)

        prior = TwoStateGaussianState(init_mean, init_cov, start_time=start_time,
                                      end_time=end_time)
        new_tracks = set()
        for detection in detections:
            hyp = SingleProbabilityHypothesis(prediction=prior, measurement=detection,
                                              probability=Probability(1.0))
            state = self.updater.update(hyp)
            track = Track([state], id=self._max_track_id)
            track.exist_prob = Probability(1)
            self._max_track_id += 1
            new_tracks.add(track)

        return new_tracks



class TwoStateMeasurementInitiator(TwoStateInitiator):

    skip_non_reversible: bool = Property(default=False)
    diag_load: float = Property(default=0.0, doc="Positive float value for diagonal loading")

    def initiate(self, detections, start_time, end_time, **kwargs):

        new_tracks = set()
        for detection in detections:
            measurement_model = detection.measurement_model

            if isinstance(measurement_model, LinearModel):
                model_matrix = measurement_model.matrix()
                inv_model_matrix = np.linalg.pinv(model_matrix)
                state_vector = inv_model_matrix @ detection.state_vector
            else:
                if isinstance(measurement_model, ReversibleModel):
                    try:
                        state_vector = measurement_model.inverse_function(detection)
                    except NotImplementedError:
                        if not self.skip_non_reversible:
                            raise
                        else:
                            continue
                    model_matrix = measurement_model.jacobian(State(state_vector))
                    inv_model_matrix = np.linalg.pinv(model_matrix)
                elif self.skip_non_reversible:
                    continue
                else:
                    raise Exception("Invalid measurement model used.\
                                    Must be instance of linear or reversible.")

            model_covar = measurement_model.covar()

            init_mean = self.prior.state_vector.copy()
            init_cov = self.prior.covar.copy()


            init_mean, init_cov = predict_state_to_two_state(init_mean, init_cov,
                                                             self.transition_model,
                                                             end_time - start_time)
            mapped_dimensions = measurement_model.mapping

            init_mean[mapped_dimensions, :] = 0
            init_cov[mapped_dimensions, :] = 0
            C0 = inv_model_matrix @ model_covar @ inv_model_matrix.T
            C0 = C0 + init_cov + np.diag(np.array([self.diag_load] * C0.shape[0]))
            if not isPD(C0):
                C0 = nearestPD(C0)
            init_mean = init_mean + state_vector
            prior = TwoStateGaussianState(init_mean, C0, start_time=start_time,
                                          end_time=end_time)
            hyp = SingleProbabilityHypothesis(prediction=prior, measurement=detection,
                                              probability=Probability(1.0))
            state = self.updater.update(hyp)
            track = Track([state], id=self._max_track_id)
            track.exist_prob = Probability(1)
            self._max_track_id += 1
            new_tracks.add(track)

        return new_tracks



