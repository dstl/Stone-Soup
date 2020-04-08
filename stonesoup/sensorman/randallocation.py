

from .base import SensorManModel
from ..base import Property
from random import shuffle
import numpy as np
from stonesoup.types.detection import Detection
from scipy.stats import multivariate_normal

class RandAllocation(SensorManModel):

    n_targets = Property(int,
                     default=5,
                     doc="Default number of target - track pairs the sensor allocation will work with.")

    sensor_target_mapping = Property(list,
                         default=[],
                         doc="Variable to hold the sensor to target mapping.")

    def allocate(self, sensorlist, **kwargs):
    # Accepts lists of sensors possibly will accept a list of tracks (somehow?) in the future?
        n_sensors = len(sensorlist)

        shuffle(sensorlist)

        self.sensor_target_mapping = [0] * self.n_targets
        for n in range(0, n_sensors):
            m = np.random.randint(0, self.n_targets)
            self.sensor_target_mapping[m] = sensor_target_mapping[m] + 1

        # Allocate specific sensors to tracks
        sensorlist_output = []
        for current_track in self.sensor_target_mapping:
            sensor_list_current_track = []
            for n in range(0, current_track):
                sensor_list_current_track.append(sensorlist.pop())
            sensorlist_output.append(sensor_list_current_track)

        #print(sensorlist_output)
        # The length of sensorlist_output in its outer most dimension should be the same as n_targets
        return sensorlist_output

    def reallocate(self, sensorlist):

        m = np.random.randint(0, self.n_targets)
        n = np.random.randint(0, self.n_targets)
        while m==n:
            n = np.random.randint(0, self.n_targets)

        sensorlist[n].append(sensorlist[m].pop())

        return sensorlist

    def generate_measurements(self, truth, sigma):

        multi_measurements = []

        for k in range(0,self.n_targets): # loops around targets
            tmp_lst2 = []
            m = 0
            for j in range(0,self.sensor_target_mapping[k]): # loops around the number of sensors allocated to that target
                tmp_lst = []
                for state in truth[k]:
                    x, y = multivariate_normal.rvs(
                        state.state_vector.ravel(), cov=np.diag([sigma[m], sigma[m]]))
                    tmp_lst.append(Detection(
                        np.array([[x], [y]]), timestamp=state.timestamp))
                tmp_lst2.append(tmp_lst)
                m += 1
            multi_measurements.append(tmp_lst2)
        return multi_measurements

