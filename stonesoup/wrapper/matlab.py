import numpy as np
import matlab
import matlab.engine

from .base import Wrapper
from ..base import Property


class MatlabWrapper(Wrapper):
    """
    Wrapper for creating/connecting to matlab sessions, running MATLAB code \
    and converting the output.

    Notes
    =====
    * Python compatibility:
        - Only MATLAB 2016b and later releases can be used with Stone Soup.
        - MATLAB 2017a and earlier versions only support up to Python 3.5.
        - MATLAB 2017b and later versions include support for Python 3.6. 
        - Support for Python 3.7 may potentially be added in MATLAB 2019a.
    * General remarks:
        - The currect version of the class is only intended for use with \
            synchronous MATLAB engine creation.
    """

    matlab_engine = Property(
        matlab.engine.MatlabEngine, default=None,
        doc="The underlying Matlab engine to be used by the wrapper. Use the \
             static class methods MatlabWrapper.start_engine() or \
             MatlabWrapper.connect_engine() to generate a compatible input. If\
             set to None on initialisation, the Wrapper will initiate a new \
             Matlab session with the default settings.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if(self.matlab_engine is None):
            self.matlab_engine = self.start_engine()
        self.matlab_engine.addpath(self.matlab_engine.genpath(
            self.dir_path, nargout=1), nargout=0)

    @staticmethod
    def start_engine(*args, **kwargs):
        """Start a new MATLAB session and return handle to an engine connected to it.

        Returns
        -------
        MatlabEngine
            Python object for communicating with MATLAB.
        """
        eng = matlab.engine.start_matlab(*args, *kwargs)
        return eng

    @staticmethod
    def connect_engine(session_name=None, *args, **kwargs):
        """Connect new engine to a running MATLAB shared session.

        Parameters
        ----------
        session_name : str, optional
            Name of the shared session. To find the name of a shared session, \
            call MatlabWrapper.find_sessions(). (the default is None, in \
            which case the new engine will connect to the first session \
            named in the tuple returned by MatlabWrapper.find_sessions())

        Returns
        -------
        MatlabEngine
            Python object for communicating with MATLAB.

        Note
        ----
        - Connecting to a shared session requires some initial setup to \
          initialise the shared connection in MATLAB. See \
          `here <https://uk.mathworks.com/help/matlab/ref/matlab.engine.shareengine.html>`_ \
          for more information.
        """
        eng = matlab.engine.connect_matlab(session_name, *args, *kwargs)
        return eng

    def stop_engine(self):
        """
        End the MATLAB engine
        """

        self.matlab_engine.quit()
        return

    @classmethod
    def matlab_array(cls, array):
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
            mat_array = cls.matlab_logical_array(array)

        elif all([type(i) == int for i in elems]):
            # Defaulting to 32bit because matlab cannot return
            #  64bit ints to python
            mat_array = cls.matlab_int32_array(array)

        elif all([type(i) == float for i in elems]):
            mat_array = cls.matlab_double_array(array)

        else:
            # When all else fails, try doubles
            mat_array = cls.matlab_double_array(array)

        return mat_array

    @classmethod
    def matlab_double_array(self, array):
        """"Converts an array or list of lists to a MATLAB double array
        """

        if isinstance(array, np.ndarray):
            array = array.tolist()

        return matlab.double(array)

    @classmethod
    def matlab_single_array(self, array):
        """"Converts an array or list of lists to a MATLAB single array
        """
        if isinstance(array, np.ndarray):
            array = array.tolist()

        return matlab.single(array)

    @classmethod
    def matlab_int32_array(self, array):
        """
        Converts an array or list of lists to a MATLAB int32 array
        """

        if isinstance(array, np.ndarray):
            array = array.tolist()

        return matlab.int32(array)

    @classmethod
    def matlab_uint32_array(self, array):
        """
        Converts an array or list of lists to a MATLAB uint32 array
        """
        if isinstance(array, np.ndarray):
            array = array.tolist()

        return matlab.uint32(array)

    @classmethod
    def matlab_logical_array(self, array):
        """
        Converts an array or list of lists to a MATLAB logical array
        """

        if isinstance(array, np.ndarray):
            array = array.tolist()

        return matlab.logical(array)
