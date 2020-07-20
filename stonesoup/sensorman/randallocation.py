from .base import SensorManModel
from ..base import Property

import numpy as np

class RandAllocation(SensorManModel):



    def reallocate(self, sensorlist):
        """
        Reallocates a sensorlist (a list of sub lists) typically called every loop iteration.

        Possibly rename in future to "update" or similar.

        """
        m = np.random.randint(0, self.n_targets)
        while len(sensorlist[m]) == 0: # Need to check if the m-th list of sensors is empty
            m = np.random.randint(0, self.n_targets)

        n = np.random.randint(0, self.n_targets)
        while m==n:
            n = np.random.randint(0, self.n_targets)

        sensorlist[n].append(sensorlist[m].pop())
        # Update target_sensor_mapping
        self.target_sensor_mapping[n].append(self.target_sensor_mapping[m].pop())

        return sensorlist
