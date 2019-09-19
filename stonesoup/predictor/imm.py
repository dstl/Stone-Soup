import numpy as np
from functools import lru_cache

from .base import Predictor
from ..base import Property, Base
from ..functions import gauss2sigma, unscented_transform, imm_merge
from ..types.prediction import (GaussianStatePrediction,
                                WeightedGaussianStatePrediction,
                                GaussianMixtureStatePrediction)
from ..types.state import (State, GaussianMixtureState, GaussianState,
                           WeightedGaussianState)


class IMMPredictor(Base):
    predictors = Property([Predictor],
                          doc="A bank of predictors each parameterised with "
                              "a different model")
    model_transition_matrix = \
        Property(np.ndarray,
                 doc="The square transition probability "
                     "matrix of size equal to the number of "
                     "predictors")

    @lru_cache()
    def predict(self, prior, control_input=None, timestamp=None, **kwargs):
        """
        IMM prediction step
        Parameters
        ----------
        prior: :class:`~GaussianMixtureState`
            The prior state
        control_input : :class:`~.State`, optional
            The control input. It will only have an effect if
            :attr:`control_model` is not `None` (the default is `None`)
        timestamp: :class:`datetime.datetime`, optional
            A timestamp signifying when the prediction is performed \
            (the default is `None`)

        Returns
        -------
        :class:`~GaussianMixtureState`
        """
        nm = self.model_transition_matrix.shape[0]

        # Extract means, covars and weights
        means = prior.means
        covars = prior.covars
        weights = prior.weights

        # Step 1) Calculation of mixing probabilities
        c_j = self.model_transition_matrix.T @ weights
        mu_ij = (self.model_transition_matrix * (weights @ (1 / c_j).T)).T

        # Step 2) Mixing (Mixture Reduction)
        means_k, covars_k = imm_merge(means, covars, mu_ij)

        # Step 3) Mode-matched prediction
        predictions = []
        for i in range(nm):
            prior_i = GaussianState(means_k[:, [i]],
                                    np.squeeze(covars_k[[i], :, :]),
                                    timestamp=prior.timestamp)
            prediction = self.predictors[i].predict(prior_i,
                                                    timestamp=timestamp)
            predictions.append(
                WeightedGaussianStatePrediction(
                      prediction.mean,
                      prediction.covar,
                      weight=weights[i, 0],
                      timestamp=prediction.timestamp))

        return GaussianMixtureStatePrediction(predictions)