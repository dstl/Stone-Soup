import sys
import numpy as np
from datetime import datetime

from stonesoup.detector import beamformers_2d
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
                                               ConstantVelocity
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.types.array import CovarianceMatrix
from stonesoup.plotter import Plotter
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.updater.kalman import KalmanUpdater
from stonesoup.types.state import GaussianState
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.track import Track
import matplotlib.pyplot as plt


# Get the arguments specified on the command line when the function was called
#
# Inputs:
#       args - string containing the command line arguments
#       flag - string corresponding to a required argument
#
# Outputs:
#       arg.split(flag)[1] - string holding the value of the argument
def get_arg(args, flag):
    found = False
    for arg in args:
        if(arg.find(flag) >= 0):
            found = True
            return arg.split(flag)[1]
    if(found is False):
        raise Exception('Required argument {} was not found' .format(flag))


args = sys.argv
data_file = get_arg(args, '--datafile=')

transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.01),
                                                          ConstantVelocity(0.01)])

covar = CovarianceMatrix(np.array([[1, 0], [0, 1]]))
measurement_model = LinearGaussian(ndim_state=4, mapping=[0, 2], noise_covar=covar)

predictor = KalmanPredictor(transition_model)

updater = KalmanUpdater(measurement_model)

prior = GaussianState([[0.5], [0], [0.5], [0]], np.diag([1, 0, 1, 0]), timestamp=datetime.now())

detector1 = beamformers_2d.capon(data_file, sensor_loc="0 0 0; 0 10 0; 0 20 0; 10 0 0; 10 10 0; 10 20 0; 20 0 0; 20 10 0; 20 20 0", fs=2000, omega=50, wave_speed=1481)
detector2 = beamformers_2d.rjmcmc(data_file, sensor_loc="0 0 0; 0 10 0; 0 20 0; 10 0 0; 10 10 0; 10 20 0; 20 0 0; 20 10 0; 20 20 0", fs=2000, omega=50, wave_speed=1481)

track1 = Track()
track2 = Track()
print("RJMCMC detections:")
for timestep, detections in detector2:
    for detection in detections:
        print(detection)
        prediction = predictor.predict(prior, timestamp=detection.timestamp)
        hypothesis = SingleHypothesis(prediction, detection)  # Group a prediction and measurement
        post = updater.update(hypothesis)
        track2.append(post)
        prior = track2[-1]
print("Capon detections:")
for timestep, detections in detector1:
    for detection in detections:
        print(detection)
        prediction = predictor.predict(prior, timestamp=detection.timestamp)
        hypothesis = SingleHypothesis(prediction, detection)  # Group a prediction and measurement
        post = updater.update(hypothesis)
        track1.append(post)
        prior = track1[-1]


plotter = Plotter()
plotter.plot_tracks(set([track1, track2]), [0, 2], uncertainty=True)
plotter.fig

plt.show()
