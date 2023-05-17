from stonesoup.predictor import Predictor
from stonesoup.predictor._utils import predict_lru_cache
from stonesoup.types.prediction import Prediction

from stonesoup.custom.functions import predict_state_to_two_state

class TwoStatePredictor(Predictor):

    @predict_lru_cache()
    def predict(self, prior, current_end_time=None, new_start_time=None, new_end_time=None,
                **kwargs):
        statedim = self.transition_model.ndim_state
        mu = prior.mean[-statedim:]
        C = prior.covar[-statedim:, -statedim:]
        if new_start_time > current_end_time:
            dt = new_start_time - current_end_time
            A = self.transition_model.matrix(time_interval=dt)
            Q = self.transition_model.covar(time_interval=dt)
            mu = A @ mu
            C = A @ C @ A.T + Q
        elif new_start_time < current_end_time:
            raise ValueError('newStartTime < currentEndTime - scan times messed up!')

        two_state_mu, two_state_cov = predict_state_to_two_state(mu, C, self.transition_model,
                                                                 new_end_time - new_start_time)

        return Prediction.from_state(prior, two_state_mu, two_state_cov,
                                     start_time=new_start_time,
                                     end_time=new_end_time)
