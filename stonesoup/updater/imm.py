# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import multivariate_normal as mvn
from functools import lru_cache

from .base import Updater
from ..base import Property, Base
from ..types.hypothesis import SingleHypothesis
from ..types.prediction import (WeightedGaussianMeasurementPrediction,
                                GaussianMixtureMeasurementPrediction)
from ..types.update import (WeightedGaussianStateUpdate,
                            GaussianMixtureStateUpdate)
from ..types.state import WeightedGaussianState


class IMMUpdater(Base):

    updaters = Property([Updater],
                        doc="A bank of predictors each parameterised with "
                            "a different model")
    model_transition_matrix = \
        Property(np.ndarray,
                 doc="The square transition probability "
                     "matrix of size equal to the number of "
                     "updaters")

    @lru_cache()
    def predict_measurement(self, predicted_state, measurement_model=None,
                            **kwargs):
        """IMM measurement prediction step

        Parameters
        ----------
        predicted_state : :class:`~.GaussianMixtureStatePrediction`
            A predicted state object

        Returns
        -------
        : :class:`~.GaussianMixtureMeasurementPrediction`
            The measurement prediction
        """
        nm = self.model_transition_matrix.shape[0]

        # Extract means, covars and weights
        means, covars, weights = (predicted_state.means,
                                  predicted_state.covars,
                                  predicted_state.weights)

        meas_predictions = []
        for i in range(nm):
            pred = WeightedGaussianState(means[:, [i]],
                                         np.squeeze(covars[[i], :, :]),
                                         timestamp=predicted_state.timestamp)
            #pred = self.updaters[i].convert2local_state(pred)
            meas_prediction = self.updaters[i].predict_measurement(pred)
            #meas_prediction = self.updaters[i].convert2common_state(
            #        meas_prediction)
            meas_predictions.append(
                WeightedGaussianMeasurementPrediction(
                    meas_prediction.mean,
                    meas_prediction.covar,
                    cross_covar=meas_prediction.cross_covar,
                    timestamp=meas_prediction.timestamp,
                    weight=weights[i, 0]))
        return GaussianMixtureMeasurementPrediction(meas_predictions)

    def update(self, hypothesis, **kwargs):
        """ IMM measurement update step

        Parameters
        ----------
        hypothesis : :class:`~.Hypothesis`
            Hypothesis with predicted state and associated detection used for
            updating.

        Returns
        -------
        : :class:`~.GaussianMixtureStateUpdate`
            The state posterior
        """
        nm = self.model_transition_matrix.shape[0]

        # Extract weights
        weights = hypothesis.prediction.weights

        # Step 3) Mode-matched filtering (ctn'd)
        Lj = []
        posteriors = []
        for i in range(nm):
            pred = hypothesis.prediction.components[i]
            pred_orig = pred
            #pred = self.updaters[i].convert2local_state(pred)
            meas_prediction = self.updaters[i].predict_measurement(pred)

            # Both the meas_prediction and pred need to be in the same format
            # i.e. local or common
            #meas_prediction = self.updaters[i].convert2common_state(
            #        meas_prediction)
            #pred = pred_orig #Saves call to convert
            hyp = SingleHypothesis(pred, hypothesis.measurement,
                                   meas_prediction)
            posterior = self.updaters[i].update(hyp)
            Lj.append(mvn.pdf(posterior.hypothesis.measurement.state_vector.T,
                              posterior.hypothesis.
                              measurement_prediction.mean.ravel(),
                              meas_prediction.covar))
            # If meas_prediction & pred are in common then we don't need to
            # convert the posterior
            # posterior = self.updaters[i].convert2common_state(posterior)
            posteriors.append(posterior)

        # Step 4) Mode Probability update
        c_j = self.model_transition_matrix.T @ weights  # (11.6.6-8)
        weights = Lj * c_j.ravel()  # (11.6.6-15)
        weights = weights / np.sum(weights)  # Normalise
        posteriors_w = [WeightedGaussianStateUpdate(
                            posteriors[i].mean,
                            posteriors[i].covar,
                            posteriors[i].hypothesis,
                            weight=weights[i],
                            timestamp=posteriors[i].timestamp)
                        for i in range(nm)]

        return GaussianMixtureStateUpdate(posteriors_w, hypothesis)
