from functools import lru_cache

import numpy as np
from scipy.stats import multivariate_normal

from ..base import Property
from ..types.prediction import (
    MeasurementPrediction,
)
from ..types.state import PointMassState
from ..types.update import Update
from .base import Updater


class PointMassUpdater(Updater):
    """Point mass Updater

    Perform an update by multiplying grid points weights by PDF of measurement
    model
    """

    sFactor: float = Property(default=4.0, doc="How many sigma to cover by the grid")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update(self, hypothesis, **kwargs):
        """Point mass update step

        Parameters
        ----------
        hypothesis : :class:`~.Hypothesis`
            Hypothesis with predicted state and associated detection used for
            updating.

        Returns
        -------
        : :class:`~.ParticleMeasurementPrediction`
            The state posterior
        """

        predicted_state = Update.from_state(
            state=hypothesis.prediction,
            hypothesis=hypothesis,
            timestamp=hypothesis.prediction.timestamp,
        )

        measurement_model = hypothesis.measurement.measurement_model

        R = measurement_model.covar()  # Noise

        x = measurement_model.function(predicted_state)  # State to measurement space
        pdf_value = multivariate_normal.pdf(
            x.T, np.ravel(hypothesis.measurement.state_vector), R
        )  # likelihood
        new_weight = np.ravel(hypothesis.prediction.weight) * np.ravel(pdf_value)

        new_weight = new_weight / (
            np.prod(hypothesis.prediction.grid_delta) * sum(new_weight)
        )  # Normalization

        predicted_state = PointMassState(
            state_vector=hypothesis.prediction.state_vector,
            weight=new_weight,
            grid_delta=hypothesis.prediction.grid_delta,
            grid_dim=hypothesis.prediction.grid_dim,
            center=hypothesis.prediction.center,
            eigVec=hypothesis.prediction.eigVec,
            Npa=hypothesis.prediction.Npa,
            timestamp=hypothesis.prediction.timestamp,
        )

        return predicted_state

    @lru_cache()
    def predict_measurement(self, state_prediction, measurement_model=None, **kwargs):

        if measurement_model is None:
            measurement_model = self.measurement_model

        new_state_vector = measurement_model.function(state_prediction, **kwargs)

        return MeasurementPrediction.from_state(
            state_prediction,
            state_vector=new_state_vector,
            timestamp=state_prediction.timestamp,
        )
