import numpy as np
from scipy.stats import mvn
import itertools
from numpy import linalg as LA
import scipy.special as sciSpec
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import fftconvolve


def boxvertex(n, bound):
    bound = np.flipud(bound)
    vertices = np.zeros((2**n, n))
    for k in range(2**n):
        for d in range(n):
            if k & (1 << d):
                vertices[k, d] = bound[d]
            else:
                vertices[k, d] = -bound[d]
    return vertices



def measPdfPrepFFT(measPdf, gridDimOld, predMeanEst, predVarEst, F, sFactor, nx, Npa, k):
    # Setup the measurement grid
    eigVal, eigVect = np.linalg.eig(predVarEst)  # eigenvalue and eigenvectors, for setting up the grid
    gridBoundWant = np.sqrt(eigVal) * sFactor  # Wanted boundaries of pred grid
    gridBoundWantCorners = np.dot(boxvertex(nx, gridBoundWant), eigVect.T).T + predMeanEst  # Wanted corner of predictive grid
    gridBoundWantCorners = np.dot(np.linalg.inv(F), gridBoundWantCorners)  # Back to filtering space
    maxF = np.max(gridBoundWantCorners, axis=1)  # Min/Max meas corners
    minF = np.min(gridBoundWantCorners, axis=1)
    gridDim = []
    gridStep = np.zeros((nx, 1))
    for ind3 in range(nx):  # Creation of filtering grid so that it creates wanted predictive grid
        gridDim.append(np.linspace(minF[ind3], maxF[ind3], Npa))
        gridStep[ind3] = abs(gridDim[ind3][0] - gridDim[ind3][1])
    measGridNew = np.array(np.meshgrid(*gridDim)).reshape(nx, -1, order='C')
  

    GridDelta = gridStep  # Grid step size
    GridDelta = np.squeeze(GridDelta)

    # Interpolation
    Fint = RegularGridInterpolator(gridDimOld, measPdf.reshape(Npa, Npa, order='F'), method="linear", bounds_error=False, fill_value=0)
    if k == 0:
        filtGridInterpInvTrsf = measGridNew.T
    else:
        filtGridInterpInvTrsf = np.dot(np.linalg.inv(F), measGridNew).T
    measPdf = Fint(filtGridInterpInvTrsf)
    
    gridDimOld = gridDim
    
    # # Unpack x, y coordinates from measGrid
    # x_coords, y_coords = measGridNew

    # # Plot the data as a scatter plot
    # plt.figure()
    # plt.scatter(x_coords, y_coords, c=measPdf, cmap='viridis')
    

    return measPdf, gridDimOld, GridDelta, measGridNew



def pmfUpdateFFT(F, measPdf, measGridNew, GridDelta, k, Npa, invQ, predDenDenomW, nx):
    # Predictive grid
    predGrid = np.dot(F, measGridNew)
    
    # Grid step size
    GridDelta[:, k+1] = np.dot(F, GridDelta[:, k])
    
    # ULTRA FAST PMF
    filtDenDOTprodDeltas = np.dot(measPdf, np.prod(GridDelta[:, k]))  # measurement PDF * measurement PDF step size
    filtDenDOTprodDeltasCub = np.reshape(filtDenDOTprodDeltas, (Npa, Npa), order='C')  # Into physical space
    
    halfGrid = (np.ceil(predGrid.shape[1] / 2)-1).astype(int)
    
    pom = np.transpose(predGrid[:, halfGrid][:, np.newaxis] - predGrid)  # Middle row of the TPM matrix
    TPMrow = (np.exp(np.sum(-0.5 * pom @ invQ * pom, axis=1)) / predDenDenomW).reshape(1, -1, order='C')  # Middle row of the TPM matrix
    TPMrowCubPom = np.reshape(TPMrow, (Npa, Npa), order='F')  # Into physical space
    
    # Compute the convolution using scipy.signal.fftconvolve
    convolution_result_complex = fftconvolve(filtDenDOTprodDeltasCub, TPMrowCubPom, mode='same')

    # Take the real part of the convolution result to get a real-valued result
    convolution_result_real = np.real(convolution_result_complex).T
    
    
    predDensityProb = np.reshape(convolution_result_real, (-1,1), order='F')
    predDensityProb = predDensityProb / (np.sum(predDensityProb) * np.prod(GridDelta[:, k+1]))  # Normalization (theoretically not needed)
    
    
    return predDensityProb, predGrid, GridDelta


def ukfUpdate(measVar,nx,kappa,measMean,ffunct,k,Q):
    # UKF prediction for grid placement
    S = np.linalg.cholesky(measVar) #lower choleski
    decomp = np.sqrt(nx+kappa)*S
    rep = np.matlib.repmat(measMean.T,2*nx,1).T + np.c_[decomp,-decomp] #concatenate
    chi = np.c_[measMean, rep]
    wUKF = np.array(np.c_[kappa,np.matlib.repmat(0.5,1,2*nx)])/(nx+kappa) #weights
    
    Y = ffunct(chi, np.zeros((nx,1)),k)
    xp_aux = Y @ wUKF.T
    Ydiff = Y - xp_aux
    Pp_aux =  np.multiply(Ydiff,np.matlib.repmat(wUKF,nx,1))@Ydiff.T+Q.T  # UKF prediction var
    return xp_aux,Pp_aux


def gridCreationFFT(xp_aux, Pp_aux, sFactor, nx, Npa):
    # Boundaries of grid
    gridBound = np.sqrt(np.diag(Pp_aux)) * sFactor 
    
    # Creation of propagated grid
    gridDim = []
    gridStep = np.zeros((nx, 1))
    for ind3 in range(nx):
        gridDim.append(np.linspace(-gridBound[ind3], gridBound[ind3], Npa) + xp_aux[ind3])
        gridStep[ind3] = abs(gridDim[ind3][0] - gridDim[ind3][1])
    
    # Grid rotation by eigenvectors and translation to the counted unscented mean
    predGrid = np.array(np.meshgrid(*gridDim)).reshape(nx, -1, order='C')
    
    # Grid step size
    predGridDelta = np.squeeze(gridStep)

    return predGrid, gridDim, predGridDelta


def gridCreation(xp_aux,Pp_aux,sFactor,nx,Npa):
    gridDim = np.zeros((nx,Npa))
    gridStep = np.zeros(nx)
    eigVal,eigVect = LA.eig(Pp_aux) # eigenvalue and eigenvectors for setting up the grid
    gridBound = np.sqrt(eigVal)*sFactor #Boundaries of grid
    
    for ind3 in range(0,nx):  #Creation of propagated grid
        gridDim[ind3] = np.linspace(-gridBound[ind3], gridBound[ind3], Npa) #New grid with middle in 0
        gridStep[ind3] = np.absolute(gridDim[ind3][0] - gridDim[ind3][1]) #Grid step
    
    combvec_predGrid = np.array(list(itertools.product(*gridDim)))
    predGrid_pom = np.dot(combvec_predGrid,eigVect).T               
    size_pom = np.size(predGrid_pom,1)
    predGrid = predGrid_pom + np.matlib.repmat(xp_aux,1,size_pom) #Grid rotation by eigenvectors and traslation to the counted unscented mean
    predGridDelta = gridStep # Grid step size
    return predGrid,predGridDelta


def pmfMeas(predGrid,nz,k,z,invR,predDenDenomV,predDensityProb,predGridDelta,hfunct):
    predThrMeasEq = hfunct(predGrid,np.zeros((nz,1)),k+1) #Prediction density grid through measurement EQ
    pom = np.matlib.repmat(z,np.size(predThrMeasEq.T,0),1)-predThrMeasEq.T  #Measurement - measurementEQ(Grid) 
    citatel = np.exp(np.sum(-0.5*np.multiply(pom @ invR,pom),1))
    filterDensityNoNorm = np.multiply(citatel / predDenDenomV ,predDensityProb.T)
    filterDensityNoNorm = filterDensityNoNorm.T
    measPdf = (filterDensityNoNorm / np.sum(np.prod(predGridDelta)*filterDensityNoNorm,0))
    return measPdf

def pmfUpdateSTD(measGrid,measPdf,predGridDelta,ffunct,predGrid,nx,k,invQ,predDenDenomW,N):
    fitDenDOTprodDeltas = measPdf*np.prod(predGridDelta[:,k]) # measurement PDF * measurement PDF step size
    gridNext = ffunct(measGrid,np.zeros((nx,1)),k+1) # Old grid through dynamics
     
    predDensityProb = np.zeros((N,1))
    for ind2 in range(0,N): #Over number of state of prediction grid
        pom = (predGrid[:,ind2].T-(gridNext).T)    
        suma = np.sum(-0.5*np.multiply(pom@invQ,pom),1)
        predDensityProb[ind2,0] = (np.exp(suma)/predDenDenomW).T@fitDenDOTprodDeltas
    predDensityProb = predDensityProb/(np.sum(predDensityProb)*np.prod(predGridDelta[:,k+1])) # Normalizaton (theoretically not needed)
    return predDensityProb


def pmfUpdateDWC(invF,predGrid,measGrid,predGridDelta,Qa,cnormHere,measPdf,N,k):
    predDensityProb = np.zeros((N,N))
    for i in range(0,N): # Unecessary for cycle, for clearer understanding
        ma = invF @ predGrid[:,i]
        for n in range(0,N):
            lowerBound = np.array([measGrid[:,n]-predGridDelta[:,k]/2]).T #  boundary of rectangular region M
            upperBound = np.array([measGrid[:,n]+predGridDelta[:,k]/2]).T
            cdfAct = mvn.mvnun(lowerBound,upperBound,ma,Qa)[0] #Integral calculation
            predDensityProb[i,n] = cnormHere*cdfAct*measPdf[n] # Predictive density
    predDensityProb = np.sum(predDensityProb,1)
    predDensityProb = predDensityProb/(np.sum(predDensityProb)*np.prod(predGridDelta[:,k+1])) # Normalizaton (theoretically not needed)
    return predDensityProb


def pmfUpdateDiagDWC(measGrid,N,predGridDelta,F,predGrid,Q,s2,measPdf,normDiagDWC,k):
    predDensityProb = np.zeros((N,N))
    bound = np.zeros((np.size(measGrid,0),np.size(measGrid,0))) # boundary of rectangular region M
    
    for i in range(0,N):
        for n in range(0,N):
            bound = np.array([measGrid[:,n]-predGridDelta[:,k]/2,measGrid[:,n]+ predGridDelta[:,k]/2]).T
            pom = np.array([ np.divide(-F@bound[:,0] + predGrid[:,i],np.sqrt(np.diag(Q))), np.divide(-F@bound[:,1] + predGrid[:,i],np.sqrt(np.diag(Q))) ]).T # NESEDI DIMENZE!!!!!!!!!!       
            erfAct = np.prod((0.5 - 0.5*sciSpec.erf(pom[:,1]/s2)) - (0.5 - 0.5*sciSpec.erf(pom[:,0]/s2)))
            predDensityProb[i,n] = measPdf[n] * normDiagDWC * erfAct # Predictive density
    predDensityProb = np.sum(predDensityProb,1)
    predDensityProb = predDensityProb/(np.sum(predDensityProb)*np.prod(predGridDelta[:,k+1]))  # Normalizaton (theoretically not needed)
    return predDensityProb



