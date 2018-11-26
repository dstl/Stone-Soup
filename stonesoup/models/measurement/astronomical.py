import ephem

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

    def observe(self,target):
        """

        :param target: a target - with a location defined in orbital elements that may be in the field of view
        :return: an observation, math:`z = [\alpha, \delta]^T` (RA and Dec), or an empty vector


        """