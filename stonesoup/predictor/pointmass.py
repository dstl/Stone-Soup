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
                wbark  = prior.weight / np.sum(prior.weight)

                # Dimensions
                s,n    = prior.state_vector.shape
                R      = np.array(np.matrix(measModel.covar()))
                ny     = R.shape[0]
                eye_s  = np.eye(s)
                
                # Predicted state mean and covariance components
                Xbark              = F @ prior.state_vector
                prior.state_vector = Xbark
                xbark              = Xbark @ wbark
                chip_              = Xbark - xbark[:, None]
                
                # Compute Ps using optimal weighting
                alpha  = 0.3 # Adjust if needed
                Ps     = alpha * (4 / (n * (s + 2)))**(2 / (s + 4)) * (chip_ * wbark) @ chip_.T + Q # Silverman's rule of thumb
                Ps     = (Ps + Ps.T) / 2
                
                # Jacobian and measurement noise
                v      = futureMeas.state_vector - measModel.function(prior)
                H      = measModel.jacobian(prior)  # Compute Jacobian at predicted state points - TODO STONE SOUP JACOBIAN NOT VECTORIZED

                # Gaussian sum update
                Ht     = np.transpose(H,[1,0,2])
                HPs    = np.einsum('ikn,kj->ijn',H,Ps)
                HPsHt  = np.einsum('ikn,kjn->ijn',HPs,Ht)
                W      = HPsHt + np.repeat(R,n).reshape(ny,ny,n)
                Winv   = np.moveaxis(np.linalg.inv(np.moveaxis(W,-1,0)),0,-1)
                PsHt   = np.einsum('ik,kjn->ijn',Ps,Ht)
                K      = np.einsum('ikn,kjn->ijn',PsHt,Winv)
                Kt     = np.transpose(K,[1,0,2])
                v      = v.reshape(ny,1,n)
                vt     = np.transpose(v,[1,0,2])
                Kv     = np.einsum('ikn,kjn->ijn',K,v)
                Kv     = Kv.reshape(s,n)
                XkGSF  = Xbark + Kv
                KH     = np.einsum('ikn,kjn->ijn',K,H)
                ImKH   = np.repeat(eye_s,n).reshape(s,s,n) - KH
                ImKHt  = np.transpose(ImKH,[1,0,2])
                PkGSF  = np.einsum('ikn,kj->ijn',ImKH,Ps)
                PkGSF  = np.einsum('ikn,kjn->ijn',PkGSF,ImKHt)
                KRK    = np.einsum('ikn,kj->ijn',K,R)
                KRK    = np.einsum('ikn,kjn->ijn',KRK,Kt)
                PkGSF += KRK
                detW   = np.moveaxis(np.linalg.det(np.moveaxis(W,-1,0)),0,-1)
                wkGSF  = np.log(wbark + np.finfo(float).eps) - np.log(np.sqrt(detW.reshape(1,n)))
                Wv     = np.einsum('ikn,kjn->ijn',Winv,v)
                vtWv   = np.einsum('ikn,kjn->ijn',vt,Wv)
                wkGSF += -0.5*vtWv.reshape(1,n)
                m      = np.max(wkGSF)
                wkGSF  = np.exp(wkGSF - (m + np.log(np.sum(np.exp(wkGSF - m)))))
                wkGSF /= np.sum(wkGSF)
                xhatk  = XkGSF @ wkGSF.T
                Phatk  = np.sum(np.multiply(PkGSF,wkGSF),axis=2)
                nuxk   = XkGSF - xhatk
                Phatk += (nuxk*wkGSF) @ nuxk.T
                Phatk  = (Phatk + Phatk.T) / 2  
    
                matrixForEig = inv(F) @ (Phatk + Q) @ inv(F.T)
                measMean     = inv(F) @ xhatk

                measGridNew, GridDeltaOld, gridDimOld, nothing, eigVect = gridCreation(
                    measMean,
                    matrixForEig,
                    self.sFactor,
                    len(invF),
                    prior.Npa,
                )
            else:
                s,n   = prior.state_vector.shape
                invFT = np.linalg.inv(F.T)
                FqF   = invF @ Q @ invFT
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
            if np.sum(predDensityProb) == 0:
                predDensityProb += 1e-120
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
