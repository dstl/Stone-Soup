import copy
from typing import Sequence

import numpy as np
from scipy.special import logsumexp
from ordered_set import OrderedSet

from .base import Predictor
from ._utils import predict_lru_cache
from .kalman import KalmanPredictor, ExtendedKalmanPredictor
from ..base import Property
from ..models.transition import TransitionModel
from ..types.prediction import Prediction
from ..types.state import GaussianState
from ..sampler import Sampler

from ..types.array import StateVectors

from numpy.linalg import inv
from numpy import linalg as LA
import itertools
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import fftconvolve
from scipy.linalg import inv
from stonesoup.types.state import PointMassState
from datetime import timedelta
import matplotlib.pyplot as plt
from stonesoup.functions import gridCreation
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default='browser'


class PointMassPredictor(Predictor):
    """ParticlePredictor class

    An implementation of a Particle Filter predictor.
    """
    sFactor: float = Property(
        default=4,
        doc="How many sigma to cover by the grid")

    #@profile
    def predict(self, prior, timestamp=None, **kwargs):
        """Particle Filter prediction step

        Parameters
        ----------
        prior : :class:`~.ParticleState`
            A prior state object
        timestamp: :class:`datetime.datetime`, optional
            A timestamp signifying when the prediction is performed
            (the default is `None`)

        Returns
        -------
        : :class:`~.ParticleStatePrediction`
            The predicted state
        """
        # Compute time_interval
        try:
            time_interval = timestamp - prior.timestamp
        except TypeError:
            time_interval = None
        
        time_difference = timedelta(days=0, hours=0, minutes=0, seconds=0)
        if time_interval == time_difference:
            predGrid = prior.state_vector,
            predDensityProb = prior.weight # SLOW LINE
            GridDelta = prior.grid_delta
            gridDimOld = prior.grid_dim
            xOld = prior.center
            Ppold = prior.eigVec
        else:
                       
            F = self.transition_model.matrix(prior=prior, time_interval=time_interval,**kwargs)
            Q = self.transition_model.covar(time_interval=time_interval, **kwargs)
            
            invF = np.linalg.inv(F)
            invFT = np.linalg.inv(F.T)
            FqF = invF@Q@invFT
            matrixForEig =  prior.covar() + FqF 
            
            measGridNew, GridDeltaOld, gridDimOld, nothing, eigVect = gridCreation(prior.mean.reshape(-1,1),matrixForEig,self.sFactor,len(invF),prior.Npa);
 
            # Interpolation
            Fint = RegularGridInterpolator(prior.grid_dim, prior.weight.reshape(prior.Npa, order='C'), method="linear", bounds_error=False, fill_value=0)
            inerpOn = np.dot(np.linalg.inv(prior.eigVec), (measGridNew - prior.center))
            measPdfNew = Fint(inerpOn.T).T
            
# =============================================================================
            # # Data for the first plot
            # toPlot1 = prior.state_vector
            # vals1 = prior.weight
            
            # # Data for the second plot
            # toPlot2 = measGridNew
            # vals2 = measPdfNew
            
            # # Create a figure
            # fig = go.Figure()
            
            # # Plot the first set of data
            # fig.add_trace(go.Scatter3d(
            #     x=toPlot1[0, :], y=toPlot1[1, :], z=vals1,
            #     mode='markers',
            #     marker=dict(size=5, color='blue', opacity=0.5),
            #     name='Measurement'
            # ))
            
            # # Plot the second set of data
            # fig.add_trace(go.Scatter3d(
            #     x=toPlot2[0, :], y=toPlot2[1, :], z=vals2,
            #     mode='markers',
            #     marker=dict(size=5, color='red', opacity=0.5),
            #     name='Interpolated'
            # ))
            
            # # Update layout for better visibility and interactivity
            # fig.update_layout(
            #     scene=dict(
            #         xaxis_title='X-axis',
            #         yaxis_title='Y-axis',
            #         zaxis_title='Values'
            #     ),
            #     title='Comparison of Prior and Measurement Data',
            #     margin=dict(l=0, r=0, t=40, b=0)  # Adjust margins for better layout
            # )
            
            # # Show the plot
            # fig.show()

# =============================================================================
            
                               
            # Predictive grid
            predGrid = np.dot(F, measGridNew)
            
            # Grid step size
            GridDelta = np.dot(F, GridDeltaOld)
            
            # ULTRA FAST PMF
            filtDenDOTprodDeltas = np.dot(measPdfNew, np.prod(GridDeltaOld))  # measurement PDF * measurement PDF step size
            filtDenDOTprodDeltasCub = np.reshape(filtDenDOTprodDeltas, prior.Npa, order='C')  # Into physical space
            
            halfGrid = (np.ceil(predGrid.shape[1] / 2)-1).astype(int)
                  
            predDenDenomW = np.sqrt((2*np.pi)**prior.ndim*np.linalg.det(Q)) #Denominator for convolution in predictive step
            
            pom = np.transpose(predGrid[:, halfGrid][:, np.newaxis] - predGrid)  # Middle row of the TPM matrix
            TPMrow = (np.exp(np.sum(-0.5 * pom @ np.linalg.inv(Q) * pom, axis=1)) / predDenDenomW).reshape(1, -1, order='C')  # Middle row of the TPM matrix
            TPMrowCubPom = np.reshape(TPMrow, prior.Npa, order='C')  # Into physical space
            
            # Compute the convolution using scipy.signal.fftconvolve
            convolution_result_complex = fftconvolve(filtDenDOTprodDeltasCub, TPMrowCubPom, mode='same')
    
            # Take the real part of the convolution result to get a real-valued result
            convolution_result_real = np.real(convolution_result_complex).T
            
            
            predDensityProb = np.reshape(convolution_result_real, (-1,1), order='F')
            predDensityProb = predDensityProb / (np.sum(predDensityProb) * np.prod(GridDelta))  # Normalization (theoretically not needed)
    
# =============================================================================
#             toPlot = predGrid
#             vals = predDensityProb
#             fig = plt.figure()
#             ax = fig.add_subplot(111, projection='3d')
#             ax.scatter(toPlot[0,:],toPlot[1,:],vals)
# =============================================================================
    
    
            xOld = F@np.vstack(prior.mean);
            Ppold = F@eigVect;
            
            # plt.plot(abs(predDensityProb))

        
        return PointMassState(state_vector=StateVectors(np.squeeze(predGrid)),
                                weight=abs(predDensityProb), # SLOW LINE
                                grid_delta = GridDelta,
                                grid_dim = gridDimOld,
                                center = xOld,
                                eigVec = Ppold,
                                Npa = prior.Npa,
                                timestamp=timestamp)


