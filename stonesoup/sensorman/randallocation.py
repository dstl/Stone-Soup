

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

    n_sensors = Property(int,
                         default=10,
                         doc="Default number of sensors.")

    sensor_target_mapping = Property(list,
                         default=[[1]],
                         doc="Variable to hold the sensor to target mapping. Default is a one-to-one mapping.")

    def create_sensor_target_mapping(self):
        self.sensor_target_mapping = [0] * self.n_targets
        for n in range(0, self.n_sensors):
            m = np.random.randint(0, self.n_targets)
            self.sensor_target_mapping[m] = self.sensor_target_mapping[m] + 1

    def allocate(self, sensorlist, **kwargs):
    # Accepts lists of sensors possibly will accept a list of tracks (somehow?) in the future?
        self.n_sensors = len(sensorlist)
        shuffle(sensorlist)
        self.create_sensor_target_mapping()

        # Allocate specific sensors to tracks
        sensorlist_output = []
        for current_track in self.sensor_target_mapping:
            sensor_list_current_track = []
            for n in range(0, current_track):
                sensor_list_current_track.append(sensorlist.pop())
            sensorlist_output.append(sensor_list_current_track)

        # print(sensorlist_output)
        # The length of sensorlist_output in its outer most dimension should be the same as n_targets
        return sensorlist_output

    def reallocate(self, sensorlist):

        m = np.random.randint(0, self.n_targets)
        while len(sensorlist[m]) == 0: # Need to check if the m-th list of sensors is empty
            m = np.random.randint(0, self.n_targets)


        n = np.random.randint(0, self.n_targets)
        while m==n:
            n = np.random.randint(0, self.n_targets)

        sensorlist[n].append(sensorlist[m].pop())
        self.sensor_target_mapping[n] += 1
        self.sensor_target_mapping[m] -= 1

        return sensorlist

    def generate_measurements(self, truth, sigma):

        multi_measurements = []

        print(self.sensor_target_mapping)
        for k in range(0, self.n_targets):  # loops around targets
            tmp_lst2 = []
            m = 0
            for j in range(0, self.sensor_target_mapping[k]): # loops around the number of sensors allocated to that target
                tmp_lst = []

                if len(sigma[k]) > 0 and type(sigma[k]) == list: # checks something is in the list
                    print(len(sigma[k]))

                    # sometimes the list of sensors 'sigma' is somehow a 3D list, where each sensor is in a list of it's own hence the check below.
                    #current_sensor_tmp =
                    if type(sigma[k][m]) == list:
                        current_sensor = sigma[k][m][0] # current_sensor_tmp[0]
                    else:
                        current_sensor = sigma[k][m]
                    #elif len(current_sensor_tmp)>0:
                    #    print('Weird list which is greater than 1?')

                    #print(current_sensor)
                    for state in truth[k]:
                        x, y = multivariate_normal.rvs(
                            state.state_vector.ravel(), cov=current_sensor.noise_covar) # sigma is a list of lists this is why it throws an error here.
                        tmp_lst.append(Detection(
                            np.array([[x], [y]]), timestamp=state.timestamp))

                tmp_lst2.append(tmp_lst)
                #print(m)
                #print(k)
                m += 1
            multi_measurements.append(tmp_lst2)
        return multi_measurements

