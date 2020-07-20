from .base import SensorManModel
from ..base import Property
from random import shuffle
import numpy as np
#from numpy import inv
#from numpy import det
from stonesoup.types.detection import Detection
from scipy.stats import multivariate_normal
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.track import Track

from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.updater.kalman import KalmanUpdater


class Greedy(SensorManModel):
    """
    Greedy sensor management algorithm.

    """

    def allocate(self, sensorlist, **kwargs):
        """

        Overloads inherited base class and makes all sensors available to all track/target pairs.

        """
        sensorlist_output = []

        # Copy the entire list of sensors to every track
        for n in range(0, self.n_targets):
            sensorlist_output.append(sensorlist)

        return sensorlist_output

    def reallocate(self, sensorlist):
        """Still to be written

        """

        sensorlist = [0, 0, 0, 0, 0]



        return sensorlist
