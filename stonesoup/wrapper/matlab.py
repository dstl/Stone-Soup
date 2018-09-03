import numpy as np
import matlab
import matlab.engine

from .base import Wrapper


class MatlabWrapper(Wrapper):
    """
    Wrapper for running MATLAB code and converting the inputs and outputs into
    suitable formats
    """

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.matlab_engine = self.start_engine()
        # Add the location of the files to be run
        self.matlab_engine.addpath(self.matlab_engine.genpath(
            self.directory_path, nargout=1), nargout=0)

    def start_engine(self):
        """
        Start the MATLAB engine for this object. By default engine is started
        on object creation
        """

        eng = matlab.engine.start_matlab()
        eng.addpath(self.directory_path, nargout=0)
        return eng

    def stop_engine(self):
        """
        End the MATLAB engine
        """

        self.matlab_engine.quit()
        return

    def matlab_array(self, array):
        """Converts a numpy array to a MATLAB array attempting to maintain the
        variable type. If the array has mixed types then array defaults to
        doubles"""

        # If it's a numpy array, convert to list; if it's anything
        #  else then throw an error
        if isinstance(array, np.ndarray):
            array = array.tolist()
        elif not isinstance(array, list):
            raise TypeError('Array must be list or numpy.ndarray to convert')

        # Get a list of all the elements in the array to check the type of
        elems = np.array(array).flatten().tolist()

        if all([type(i) == bool for i in elems]):
            mat_array = self.matlab_logical_array(array)

        elif all([type(i) == int for i in elems]):
            # Defaulting to 32bit because matlab cannot return
            #  64bit ints to python
            mat_array = self.matlab_int32_array(array)

        elif all([type(i) == float for i in elems]):
            mat_array = self.matlab_double_array(array)

        else:
            # When all else fails, try doubles
            mat_array = self.matlab_double_array(array)

        return mat_array

    def matlab_double_array(self, array):
        """"Converts an array or list of lists to a MATLAB double array
        """

        if isinstance(array, np.ndarray):
            array = array.tolist()

        return matlab.double(array)

    def matlab_single_array(self, array):
        """"Converts an array or list of lists to a MATLAB single array
        """
        if isinstance(array, np.ndarray):
            array = array.tolist()

        return matlab.single(array)

    def matlab_int32_array(self, array):
        """
        Converts an array or list of lists to a MATLAB int32 array
        """

        if isinstance(array, np.ndarray):
            array = array.tolist()

        return matlab.int32(array)

    def matlab_uint32_array(self, array):
        """
        Converts an array or list of lists to a MATLAB uint32 array
        """
        if isinstance(array, np.ndarray):
            array = array.tolist()

        return matlab.uint32(array)

    def matlab_logical_array(self, array):
        """
        Converts an array or list of lists to a MATLAB logical array
        """

        if isinstance(array, np.ndarray):
            array = array.tolist()

        return matlab.logical(array)
