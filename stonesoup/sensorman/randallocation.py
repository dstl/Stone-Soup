

from .base import SensorManModel
from ..base import Property
from random import shuffle
import numpy as np

class RandAllocation(SensorManModel):

    n_targets = Property(int,
                     default=5,
                     doc="Default number of target - track pairs the sensor allocation will work with.")

    def allocate(self, sensorlist, **kwargs):
    # Accepts lists of sensors possibly will accept a list of tracks (somehow?) in the future?

        n_sensors = len(sensorlist)



        shuffle(sensorlist)

        sensor_target_mapping = [0] * self.n_targets
        for n in range(0, n_sensors):
            m = np.random.randint(0, self.n_targets)
            sensor_target_mapping[m] = sensor_target_mapping[m] + 1


        # Allocate specific sensors to tracks
        sensorlist_output = []
        for current_track in sensor_target_mapping:
            sensor_list_current_track = []
            for n in range(0, current_track):
                sensor_list_current_track.append(sensorlist.pop())
            sensorlist_output.append(sensor_list_current_track)

        #print(sensorlist_output)
        # The length of sensorlist_output in its outer most dimension should be the same as n_targets
        return sensorlist_output
