from datetime import timedelta

import numpy as np
import plotly.io as pio
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import fftconvolve
from stonesoup.functions import grid_creation
from stonesoup.types.state import PointMassState

from ..base import Property
from ..types.array import StateVectors
from .base import Predictor

pio.renderers.default = "browser"


class PointMassPredictor(Predictor):
    """PointMassPredictor class

    An implementation of a Point Mass Filter predictor.
    """

    sFactor: float = Property(default=4., doc="How many sigma to cover by the grid")

    def predict(self, prior, timestamp=None, **kwargs):
        """Point Mass Filter prediction step

        Parameters
        ----------
        prior : :class:`~.Point mass state`
            A prior state object
        timestamp: :class:`datetime.datetime`, optional
            A timestamp signifying when the prediction is performed

        Returns
        -------
        : :class:`~.PointMassStatePrediction`
            The predicted state
        """
        # Compute time_interval
        time_interval = timestamp - prior.timestamp

        time_difference = timedelta(days=0, hours=0, minutes=0, seconds=0)
        if time_interval == time_difference:
            predGrid = (prior.state_vector,)
            predDensityProb = prior.weight
            GridDelta = prior.grid_delta
            gridDimOld = prior.grid_dim
            xOld = prior.center
            Ppold = prior.eigVec
        else:

            F = self.transition_model.matrix(
                prior=prior, time_interval=time_interval, **kwargs
            )
            Q = self.transition_model.covar(time_interval=time_interval, **kwargs)

            invF = np.linalg.inv(F)
            invFT = np.linalg.inv(F.T)
            FqF = invF @ Q @ invFT
            matrixForEig = prior.covar() + FqF

            measGridNew, GridDeltaOld, gridDimOld, nothing, eigVect = grid_creation(
                prior.mean.reshape(-1, 1),
                matrixForEig,
                self.sFactor,
                len(invF),
                prior.Npa,
            )

            # Interpolation
            Fint = RegularGridInterpolator(
                prior.grid_dim,
                prior.weight.reshape(prior.Npa, order="C"),
                method="linear",
                bounds_error=False,
                fill_value=0,
            )
            inerpOn = np.dot(np.linalg.inv(prior.eigVec), (measGridNew - prior.center))
            measPdfNew = Fint(inerpOn.T).T

            # Predictive grid
            predGrid = np.dot(F, measGridNew)

            # Grid step size
            GridDelta = np.dot(F, GridDeltaOld)

            # ULTRA FAST PMF
            # measurement PDF * measurement PDF step size
            filtDenDOTprodDeltas = np.dot(measPdfNew, np.prod(GridDeltaOld))
            filtDenDOTprodDeltasCub = np.reshape(
                filtDenDOTprodDeltas, prior.Npa, order="C"
            )  # Into physical space

            halfGrid = (np.ceil(predGrid.shape[1] / 2) - 1).astype(int)

            # Denominator for convolution in predictive step
            predDenDenomW = np.sqrt((2 * np.pi) ** prior.ndim * np.linalg.det(Q))

            pom = np.transpose(
                predGrid[:, halfGrid][:, np.newaxis] - predGrid
            )  # Middle row of the TPM matrix
            TPMrow = (
                np.exp(np.sum(-0.5 * pom @ np.linalg.inv(Q) * pom, axis=1))
                / predDenDenomW
            ).reshape(
                1, -1, order="C"
            )  # Middle row of the TPM matrix
            TPMrowCubPom = np.reshape(
                TPMrow, prior.Npa, order="C"
            )  # Into physical space

            # Compute the convolution using scipy.signal.fftconvolve
            convolution_result_complex = fftconvolve(
                filtDenDOTprodDeltasCub, TPMrowCubPom, mode="same"
            )

            # Take the real part of the convolution result to get a real-valued result
            convolution_result_real = np.real(convolution_result_complex).T

            predDensityProb = np.reshape(convolution_result_real, (-1, 1), order="F")
            # Normalization (theoretically not needed)
            predDensityProb = predDensityProb / (
                np.sum(predDensityProb) * np.prod(GridDelta)
            )

            xOld = F @ np.vstack(prior.mean)
            Ppold = F @ eigVect

        return PointMassState(
            state_vector=StateVectors(np.squeeze(predGrid)),
            weight=abs(predDensityProb),
            grid_delta=GridDelta,
            grid_dim=gridDimOld,
            center=xOld,
            eigVec=Ppold,
            Npa=prior.Npa,
            timestamp=timestamp,
        )
