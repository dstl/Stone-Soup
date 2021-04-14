# -*- coding: utf-8 -*-

import numpy as np

from stonesoup.types.update import Update
from stonesoup.updater.kalman import KalmanUpdater
from ..types.prediction import MeasurementPrediction


class ClassificationUpdater(KalmanUpdater):
    r"""Models the update step of a forward algorithm as such:

    .. math::
        X_k^i &= P(\phi^i, k) = P(\phi^i, k | y^j, k)P(y^j, k)P(\phi^i, k | \phi^l, k-1)P(\phi^l)\\
                       &= E^{ij}Z_k^jF^{il}X_{k-1}^l\\
                       &= EZ*FX_{k-1}

    Where:
    *:math:`X_k^i` is the :math:`i`-th component of the posterior state vector, representing the
    probability :math:`P(\phi^i, k)` that the state is class :math:`i` at 'time' :math:`k`, with
    the possibility of the state being any of a finite, discrete set of classes
    :math:`{\phi_i | i \in \Z_{>0}}
    *:math:`:math:`y_j` is the :math:`j`-th component of the measurement vector :math:`Z`, an
    observation of the state, assumed to be defining a multinomial distribution over the discrete,
    finite space of possible measurement classes :math:`{y_j | j \in \Z_{>0}}
    *:math:`E` defines the time invariant emission matrix of the corresponding measurement model
    *:math:`F` is the stochastic matrix defining the time invariant class transition
    :math:`P(\phi^i, k | \phi^l, k-1)
    * :math:`k` is the 'time' of update, attained from the 'time' at which the measurement
    :math:`Z` has been received
    """

    def _check_measurement_model(self, measurement_model):

        if measurement_model is None:
            if self.measurement_model is None:
                raise ValueError("No measurement model specified")
            else:
                measurement_model = self.measurement_model

        return measurement_model

    def predict_measurement(self, predicted_state, measurement_model=None, **kwargs):

        measurement_model = self._check_measurement_model(measurement_model)

        pred_meas = measurement_model.function(predicted_state, **kwargs)

        return MeasurementPrediction.from_state(predicted_state, pred_meas)

    def update(self, hypothesis, **kwargs):

        prediction = hypothesis.prediction

        try:
            E = hypothesis.measurement.measurement_model.emission_matrix
        except AttributeError:
            raise ValueError("All detections must be from class observer measurement models")

        EY = E @ hypothesis.measurement.state_vector

        prenormalise = np.multiply(prediction.state_vector, EY)

        normalise = prenormalise / np.sum(prenormalise)

        return Update.from_state(hypothesis.prediction, normalise,
                                 timestamp=hypothesis.measurement.timestamp, hypothesis=hypothesis)
