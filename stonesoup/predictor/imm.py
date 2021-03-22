import numpy as np
from functools import lru_cache
from typing import MutableSequence, List

from .base import Predictor
from ..base import Property, Base
from ..functions import gauss2sigma, unscented_transform, imm_merge
from ..types.prediction import (GaussianStatePrediction,
                                WeightedGaussianStatePrediction,
                                GaussianMixtureStatePrediction)
from ..types.state import (State, GaussianMixtureState, GaussianState,
                           WeightedGaussianState)


def null_convert(state):
    """
    Routine to do a null conversion on the Gaussian state
    Parameters
    ----------
    state: :class:'~GaussianState'
        The input state.

    Returns
    -------
    :class:'~GaussianState'
    """
    return state


class IMMPredictor(Base):
    predictors: List[Predictor] = Property( \
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
            # Convert the state from the Gaussian mixture format to the format
            # needed by the predictor
            prior_i2 = self.predictors[i].convert2local_state(prior_i)
            prediction = self.predictors[i].predict(prior_i,
                                                    timestamp=timestamp)
            # Convert the prediction format to the format need for the
            # Gaussian mixture
            #prediction = self.predictors[i].convert2common_state(prediction)
            predictions.append(
                WeightedGaussianStatePrediction(
                      prediction.mean,
                      prediction.covar,
                      weight=weights[i, 0],
                      timestamp=prediction.timestamp))

        return GaussianMixtureStatePrediction(predictions)
