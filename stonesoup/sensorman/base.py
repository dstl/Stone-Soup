# -*- coding: utf-8 -*-
from abc import abstractmethod
from ..base import Base
import scipy as sp
import numpy as np
from scipy.stats import multivariate_normal
from stonesoup.types.detection import Detection
from stonesoup.models.base import Model
from stonesoup.base import Property
from random import shuffle

class SensorManModel(Base):
    """Sensor management Model base class"""

    #ndim_state = Property(int, doc="Number of state dimensions")
    #mapping = Property(
    #    sp.ndarray, doc="Mapping between measurement and state dims")

    @property
    def ndim(self):
        raise NotImplementedError

    #@property
    #@abstractmethod
    #def ndim_meas(self):
    #    """Number of measurement dimensions"""
    #    raise NotImplementedError

    n_targets = Property(int,
                         default=5,
                         doc="Default number of target - track pairs the sensor allocation will "
                             "work with.")

    n_sensors = Property(int,
                         default=10,
                         doc="Default number of sensors.")

    target_sensor_mapping = Property(list,
                                     default=[[1]],
                                     doc="Variable to hold the sensor to target mapping. Default "
                                         "is a one-to-one mapping.")

    def create_target_sensor_mapping(self):
        """
        Creates a random target to sensor mapping. Used for initialisation.
        :return:
        """
        self.target_sensor_mapping = []
        for m in range(0, self.n_targets):
            self.target_sensor_mapping.append([])
        # think of 'n' as the sensor id that you're assigning to a random track (m)
        for n in range(0, self.n_sensors):
            m = np.random.randint(0, self.n_targets)
            self.target_sensor_mapping[m].append(n)


    def generate_measurements(self, truth, sigma):
        """
        Generates measurements according to the sensor which you have available.

        Typically this method should be employed on each timestep so that an updated sensor
        configuration can generate new measurements for whichever tracking algorithm is currently
        being employed.

        This method generates measurements for all time steps.

        """
        multi_measurements = []
        for k in range(0, len(sigma)):
            multi_measurements.append([])

        for k, track in enumerate(sigma):  # loops around targets
            m = 0
            for j, current_sensor in enumerate(track): # loops around the number of sensors allocated to that target
                tmp_lst = []

                if type(sigma[k][m]) == list:
                    current_sensor = current_sensor[0] # current_sensor_tmp[0]
                else:
                    current_sensor = current_sensor

                for state in truth[k]:
                    x, y = multivariate_normal.rvs(
                        state.state_vector.ravel(),
                        cov=current_sensor.noise_covar)  # sigma is a list of lists this is why it throws an error here.
                    tmp_lst.append(Detection(
                        np.array([[x], [y]]), timestamp=state.timestamp))

                # if len(sigma[k]) > 0 and type(sigma[k]) == list: # checks something is in the list
                #     # sometimes the list of sensors 'sigma' is somehow a 3D list, where each sensor is in a list of it's own hence the check below.
                #     #current_sensor_tmp =
                #

                multi_measurements[k].append(tmp_lst)
                m += 1

        return multi_measurements

    def allocate(self, sensorlist, **kwargs):
        """
        Accepts lists of sensors possibly will accept a list of tracks (somehow?) in the future?

        This initialises the sensor list for the sensor scheme.

        """
        local_sensorlist = sensorlist.copy()
        self.n_sensors = len(local_sensorlist)
        shuffle(local_sensorlist)
        self.create_target_sensor_mapping()

        # Allocate specific sensors to tracks
        sensorlist_output = []
        for current_track in self.target_sensor_mapping:
            sensor_list_current_track = []
            for n in range(0, len(current_track)):
                sensor_list_current_track.append(local_sensorlist.pop())
            sensorlist_output.append(sensor_list_current_track)

        # The length of sensorlist_output in its outer most dimension should be the same as n_targets
        return sensorlist_output
