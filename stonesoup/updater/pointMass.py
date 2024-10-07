
import numpy as np
from scipy.stats import multivariate_normal
from stonesoup.types.state import PointMassState
from ..base import Property
from ..types.update import Update
from .base import Updater


class PointMassUpdater(Updater):
    """Point mass Updater

    Perform an update by multiplying grid points weights by PDF of measurement
    model
    """
    sFactor: float = Property(default=4, doc="How many sigma to cover by the grid")

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
        : :class:`~.PointMassState`
            The state posterior
        """

        predicted_state = Update.from_state(
            state=hypothesis.prediction,
            hypothesis=hypothesis,
            timestamp=hypothesis.prediction.timestamp,
        )

        measurement_model = hypothesis.measurement.measurement_model

        R = measurement_model.covar()  # Noise

        x = measurement_model.function(
            predicted_state
        )  # State to measurement space
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
