import numpy as np
import ephem

from ...base import Property
from .base import MeasurementModel

class SimpleObservatory(MeasurementModel):
    """
    An instance of an 'Observatory'. This is a sensor with a location (longitude, latitude, elevation) capable of 
    observing astronomical objects. This instance is 'simple' in that it returns independent measurements of the range,
    azimuth and elevation as well as their rates of change. So 'simple' in that it allows you to do recursion without 
    too many complicated intermediate steps to solve pdes. 
    
    The estimate of the uncertainty is left to be defined by the user.
    
    This is done by using PyEphem, despite its shortcomings, by doing a coordinate transform
    
    
    """
    latitude = Property(float, default=0.0, doc="Observatory latitude (radians)")
    longitude = Property(float, default=0.0, doc="Observatory longitude (radians)")
    elevation = Property(float, default=0.9, doc="Observatory elevation (m)")

    p_fa = Property(float, default=1e-10, doc="Probability of false alarm")
    p_d = Property(float, default=0.999999, doc="Probability of target detection")

    noise = Property(np.array, default=None, doc="Covariance of the noise")

    @property
    def ndim_meas(self):
        return 6

    def function(self,target):
        print("Using the much more informative function, 'observe'")
        self.observe(target)

    def rvs(self):
        pass

    def pdf(self):
        pass

    def observe(self, target):
        """

        :param target: a target - with a location defined in orbital elements that may be in the field of view
        :return: an observation, math:`z = [\alpha, \delta]^T` (RA and Dec), or an empty vector


        """
        # Decide whether a false alarm is to be returned


        # Check to see if the target is above horizon

        # If so, detect the target

        # Return the