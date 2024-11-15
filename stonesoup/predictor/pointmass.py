from datetime import timedelta

import numpy as np
import plotly.io as pio
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import fftconvolve
from stonesoup.functions import gridCreation
from stonesoup.types.state import PointMassState

from ..base import Property
from ..types.array import StateVectors
from .base import Predictor
from scipy.linalg import inv, sqrtm, eigh

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
        
        # -------------------------------------------------------------------------------
        runGSFversion = kwargs.get("runGSFversion", False)
        
        if runGSFversion:
            futureMeas = kwargs.get("futureMeas","None")
            measModel =  kwargs.get("measModel","None")
        # -------------------------------------------------------------------------------
        
        # Compute time_interval
        time_interval = timestamp - prior.timestamp

        time_difference = timedelta(days=0, hours=0, minutes=0, seconds=0)
        if time_interval == time_difference:
            predGrid = prior.state_vector
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
            
            # -------------------------------------------------------------------------------
            # # Initialize and normalize
            if runGSFversion:
            
                # Normalize weights
                wbark = prior.weight / np.sum(prior.weight)
                
                # Predicted state mean and covariance components
                Xbark = F @ prior.state_vector
                prior.state_vector = Xbark
                s, n = Xbark.shape
                xbark = Xbark @ wbark
                chip_ = Xbark - xbark[:, None]
                
                # Compute Ps using optimal weighting
                alpha = 1  # Adjust if needed
                Ps = alpha * (4 / (n * (s + 2)))**(2 / (s + 4)) * (chip_ * wbark) @ chip_.T + Q
                Ps = (Ps + Ps.T) / 2
                
                # Jacobian and measurement noise
                H = measModel.jacobian(prior)  # Compute Jacobian at predicted state points - TODO STONE SOUP JACOBIAN NOT VECTORIZED
                Ht = np.transpose(H, (1, 0, 2))
                 

                #W = np.einsum('mnr,nd,nmr->r', H, Ps, Ht) + measModel.covar() # W = H * Ps * Ht + R
                #K = np.einsum('mnr,r->mnr', np.einsum('mn,ndr->mdr', Ps, Ht), 1/W)  # K = Ps * Ht / W - TODO some lines works only for nz = 1
                
                
                
                W = np.einsum('bar,mn,nkr->bkr', H, Ps, Ht) + measModel.covar()[:, :, np.newaxis]
                # Assuming all matrices in the array are 2x2, preallocating the inverted array
                invW = np.empty_like(W)
                # Inverting matrices using vectorized operations
                a, b, c, d = W[:, 0, 0], W[:, 0, 1], W[:, 1, 0], W[:, 1, 1]
                det = a * d - b * c  # Determinant for 2x2 matrices
                invW[:, 0, 0], invW[:, 0, 1] = d / det, -b / det
                invW[:, 1, 0], invW[:, 1, 1] = -c / det, a / det
                K = np.einsum('mn,abr,bjr->ajr', Ps, Ht, invW)

                
                # Measurement residual and update
                v = futureMeas.state_vector - measModel.function(prior)
                XkGSF = Xbark + np.einsum('abr,br->ar', K, v).reshape(K.shape[0], -1)
                
                # Posterior covariance update
                KH = np.einsum('mnr,ndr->mdr', K, H)
                eye_s = np.eye(s)
                Kt = np.transpose(K, (1, 0, 2))
                PkGSF = np.einsum('ik,kjl->kil', Ps, eye_s[:, :, None] - KH) + np.einsum('klr,ldr->kdr', K, Kt)
                
                
                # Weight update - TODO
                wkGSF = np.log(wbark) - np.log(np.sqrt(W)) + v*(v/W)
                m = np.max(wkGSF)
                wkGSF = np.exp(wkGSF - (m + np.log(np.sum(np.exp(wkGSF - m)))));
                wkGSF = wkGSF / np.sum(wkGSF);
                
                # Compute state estimate and covariance
                xhatk = XkGSF @ wkGSF.T      
                Phatk =  np.sum(PkGSF * wkGSF.reshape(1, 1, -1),axis=2)
                nuxk = XkGSF - xhatk
                
                Phatk += (nuxk * wkGSF).dot(nuxk.T)
                Phatk = (Phatk + Phatk.T) / 2  # Symmetrize
    
                matrixForEig = inv(F)@(Phatk + Q)@inv(F.T)
                measMean = inv(F)@xhatk;
                
                measGridNew, GridDeltaOld, gridDimOld, nothing, eigVect = gridCreation(
                    measMean,
                    matrixForEig,
                    self.sFactor,
                    len(invF),
                    prior.Npa,
                )
            else:
                invFT = np.linalg.inv(F.T)
                FqF = invF @ Q @ invFT
                matrixForEig = prior.covar() + FqF
                measGridNew, GridDeltaOld, gridDimOld, nothing, eigVect = gridCreation(
                    prior.mean.reshape(-1, 1),
                    matrixForEig,
                    self.sFactor,
                    len(invF),
                    prior.Npa,
                )
            # ----------------------------------------------------------------------------------------


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
                  
            Ppold = F @ eigVect;
            # ----------------------------------------------------------------------------------------
            if runGSFversion:
                xOld = xhatk;
            else:
                xOld = F @ np.vstack(prior.mean)
            # ----------------------------------------------------------------------------------------

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
