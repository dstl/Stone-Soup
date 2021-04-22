import sys
import numpy as np
from datetime import datetime

from stonesoup.detector import beamformers_2d
from stonesoup.types import state
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
                                               ConstantVelocity
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.types.array import CovarianceMatrix
from stonesoup.plotter import Plotter


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
    if(found == False):
            raise Exception('Required argument {} was not found' .format(flag))

args = sys.argv
data_file = get_arg(args, '--datafile=')

transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.01),
                                                          ConstantVelocity(0.01)])
                                                          
covar = CovarianceMatrix(np.array([[1,0],[0,1]])) # [[AA, AF],[AF, FF]]
measurement_model = LinearGaussian(ndim_state=4, mapping=[0, 2],
                       noise_covar=covar)

from stonesoup.predictor.kalman import KalmanPredictor
predictor = KalmanPredictor(transition_model)

from stonesoup.updater.kalman import KalmanUpdater
updater = KalmanUpdater(measurement_model)

from stonesoup.types.state import GaussianState
prior = GaussianState([[0.5], [0], [0.5], [0]], np.diag([1, 0, 1, 0]), timestamp=datetime.now())

detector1 = beamformers_2d.capon(data_file)

from stonesoup.types.hypothesis import SingleHypothesis

from stonesoup.types.track import Track
track = Track()

for timestep, detections in detector1:
    for detection in detections:
        print(detection)
        prediction = predictor.predict(prior, timestamp=detection.timestamp)
        hypothesis = SingleHypothesis(prediction, detection)  # Group a prediction and measurement
        post = updater.update(hypothesis)
        track.append(post)
        prior = track[-1]

plotter = Plotter()
plotter.plot_tracks(track, [0, 2], uncertainty=True)
plotter.fig

import matplotlib.pyplot as plt
plt.show()
