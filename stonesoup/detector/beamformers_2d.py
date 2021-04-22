import numpy as np
import math
import cmath
from datetime import datetime, timedelta
from stonesoup.base import Property, Base
from stonesoup.buffered_generator import BufferedGenerator
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.types.array import StateVector, CovarianceMatrix
from stonesoup.types.detection import Detection

class capon(Base, BufferedGenerator):
    csv_path: str = Property(doc='The path to the csv file, containing the raw data')
    
    @BufferedGenerator.generator_method
    def detections_gen(self):
        detections = set()
        previous_time = datetime.now()
        
        y = np.loadtxt(self.csv_path, delimiter=',')

        L = len(y)
        
        # frequency of sinusoidal signal
        omega = 50
        
        window = 20000
        windowm1 = window-1
                
        thetavals = np.linspace(0,2*math.pi, num=400)
        phivals = np.linspace(0,math.pi/2, num=100)

        # spatial locations of hydrophones
        z = np.matrix('0 0 0; 0 10 0; 0 20 0; 10 0 0; 10 10 0; 10 20 0; 20 0 0; 20 10 0; 20 20 0');
        
        N = 9 # No. of hydrophones

        # steering vector
        v = np.zeros(N, dtype=np.complex)

        # directional unit vector
        a = np.zeros(3)

        scans = []
        
        winstarts = np.linspace(0, L-window, num=int(L/window), dtype=int)
        
        c = 1481/(2*omega*math.pi)
        
        for t in winstarts:
            # calculate covariance estimate
            R = np.matmul(np.transpose(y[t:t+windowm1]), y[t:t+windowm1])
            R_inv = np.linalg.inv(R)
    
            maxF = 0
            maxtheta = 0
            maxfreq = 0
            
            for theta in thetavals:
                for phi in phivals:
                    # convert from spherical polar coordinates to cartesian
                    a[0] = math.cos(theta)*math.sin(phi)
                    a[1] = math.sin(theta)*math.sin(phi)
                    a[2] = math.cos(phi)
                    a=a/math.sqrt(np.sum(a*a));
                    for n in range(0,N):
                        phase = np.sum(a*np.transpose(z[n,]))/c
                        v[n] = math.cos(phase) - math.sin(phase)*1j
                    F = 1/((window-N)*np.transpose(np.conj(v))@R_inv@v)
                    if F > maxF:
                        maxF = F
                        maxtheta = theta
                        maxphi = phi
            
            # Defining a detection
            state_vector = StateVector([maxtheta, maxphi])  # [Azimuth, Elevation]
            covar = CovarianceMatrix(np.array([[1,0],[0,1]])) # [[AA, AE],[AE, EE]]
            measurement_model = LinearGaussian(ndim_state=4, mapping=[0, 2],
                                               noise_covar=covar)
            current_time = previous_time + timedelta(milliseconds=window)
            detection = Detection(state_vector, timestamp=current_time,
                                  measurement_model=measurement_model)
            detections = set([detection])

            scans.append((current_time, detections))

        # For every timestep
        for scan in scans:
                yield scan[0], scan[1]
        