from functools import lru_cache
from typing import Callable
import numpy as np
from scipy.stats import multivariate_normal
from stonesoup.types.state import PointMassState
from ..base import Property
from ..regulariser import Regulariser
from ..resampler import Resampler
from ..types.prediction import (
    MeasurementPrediction,
)
from ..types.update import Update
from .base import Updater


class PointMassUpdater(Updater):
    """Particle Updater

    Perform an update by multiplying particle weights by PDF of measurement
    model (either :attr:`~.Detection.measurement_model` or
    :attr:`measurement_model`), and normalising the weights. If provided, a
    :attr:`resampler` will be used to take a new sample of particles (this is
    called every time, and it's up to the resampler to decide if resampling is
    required).
    """

    sFactor: float = Property(default=3, doc="How many sigma to cover by the grid")
    resampler: Resampler = Property(
        default=None, doc="Resampler to prevent particle degeneracy"
    )
    regulariser: Regulariser = Property(
        default=None,
        doc="Regulariser to prevent particle impoverishment. The regulariser "
        "is normally used after resampling. If a :class:`~.Resampler` is defined, "
        "then regularisation will only take place if the particles have been "
        "resampled. If the :class:`~.Resampler` is not defined but a "
        ":class:`~.Regulariser` is, then regularisation will be conducted under the "
        "assumption that the user intends for this to occur.",
    )

    constraint_func: Callable = Property(
        default=None,
        doc="Callable, user defined function for applying "
        "constraints to the states. This is done by setting the weights "
        "of particles to 0 for particles that are not correctly constrained. "
        "This function provides indices of the unconstrained particles and "
        "should accept a :class:`~.ParticleState` object and return an array-like "
        "object of logical indices. ",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # @profile
    def update(self, hypothesis, **kwargs):
        """Point mass update step

        Parameters
        ----------
        hypothesis : :class:`~.Hypothesis`
            Hypothesis with predicted state and associated detection used for
            updating.

        Returns
        -------
        : :class:`~.ParticleState`
            The state posterior
        """

        predicted_state = Update.from_state(
            state=hypothesis.prediction,
            hypothesis=hypothesis,
            timestamp=hypothesis.prediction.timestamp,
        )

        measurement_model = hypothesis.measurement.measurement_model

        R = measurement_model.covar() # Noise

        x = measurement_model.function(
            predicted_state
        ) # State to measurement space
        pdf_value = multivariate_normal.pdf(
            x.T, np.ravel(hypothesis.measurement.state_vector), R
        ) # likelihood
        new_weight = np.ravel(hypothesis.prediction.weight) * np.ravel(pdf_value)

        new_weight = new_weight / (
            np.prod(hypothesis.prediction.grid_delta) * sum(new_weight)
        ) # Normalization

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
