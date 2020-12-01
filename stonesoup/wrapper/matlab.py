try:
    import matlab
    import matlab.engine
except ImportError as error:
    raise ImportError(
        "Usage of the MatlabWrapper class requires that MATLAB and the MATLAB Engine API for "
        "Python are installed. More information can be found here: "
        "https://uk.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-"
        "python.html")\
        from error

import numpy as np

from .base import Wrapper
from ..base import Property


class MatlabWrapper(Wrapper):
    """ Wrapper for creating/connecting to MATLAB sessions, running MATLAB code and converting the 
    output.

    The class makes use of the `MATLAB Engine API for Python <https://uk.mathworks.com/help/matlab/matlab-engine-for-python.html>`__
    and therefore requires that MATLAB and the engine API Python package are installed. Installation 
    instructions can be found `here <https://uk.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html>`__.
    
    To call MATLAB functions in Stone Soup as part of an object, the object needs to inherit from 
    the :class:`~.MatlabWrapper` object. All objects derived from this class need to be linked to a
    running MATLAB Engine. This is done by providing such an engine on instantiation (see 
    :attr:`~.MatlabWrapper.matlab_engine`), else an new MATLAB engine is started and remains 
    active. To call a function in the created object use:
    
    .. code :: 
    
        output_1, … , output_n = self.matlab_engine.<function name>(argument1, agument2, … , nargout = n)
    
    where ``nargout`` defines the number of expected arguments to be returned (can be zero). Each 
    of the arguments to the function needs to be in the form of a matlab array which can be created 
    from a list or numpy array using :meth:`~.MatlabWrapper.matlab_array`. The type of the output 
    array depends on the type of the input array (e.g. and array of integers is converted to a 
    matlab integer array). See `here <https://uk.mathworks.com/help/matlab/matlab_external/call-matlab-functions-from-python.html>`__
    for more details on calling MATLAB functions from Python.

    Notes
    =====
    * Python compatibility:
        - Only MATLAB 2016b and later releases can be used with Stone Soup.
        - MATLAB 2017a and earlier versions only support up to Python 3.5.
        - MATLAB 2017b and later versions include support for Python 3.6.
        - MATLAB 2019a and later versions include support for Python 3.7.
    * General remarks:
        - The currect version of the class is only intended for use with \
            synchronous MATLAB engine creation.
    """  # noqa

    matlab_engine = Property(
        matlab.engine.MatlabEngine, default=None,
        doc="The underlying engine for communicating with MATLAB. Use the static class methods "
            ":meth:`~.MatlabWrapper.start_engine` or :meth:`~.MatlabWrapper.connect_engine` to "
            "generate a compatible input. If set to ``None``, the Wrapper will initiate a new \
             Matlab engine with the default settings.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.matlab_engine is None:
            self.matlab_engine = self.start_engine()
        self.matlab_engine.addpath(self.matlab_engine.genpath(self.dir_path, nargout=1), nargout=0)

    @staticmethod
    def start_engine(*args, **kwargs):
        """Start a new MATLAB session and return handle to an engine connected to it.

        Parameters
        ----------
        *args : 
            Positional arguments, as documented by the `MATLAB documentation for 
            matlab.engine.start_matlab <https://uk.mathworks.com/help/matlab/apiref/matlab.engine.start_matlab.html>`__.
        **kwargs :
            Keyword arguments, as documented by the `MATLAB documentation for 
            matlab.engine.start_matlab <https://uk.mathworks.com/help/matlab/apiref/matlab.engine.start_matlab.html>`__.

        Returns
        -------
        :class:`matlab.engine.MatlabEngine`
            Python object for communicating with MATLAB.
        """  # noqa

        return matlab.engine.start_matlab(*args, *kwargs)

    @staticmethod
    def connect_engine(name=None, **kwargs):
        """Connect new engine to a running MATLAB shared session and return engine handle.

        Parameters
        ----------
        name : :class:`str`, optional
            Name of the shared session. To find the name of a shared session, call 
            :meth:`~.MatlabWrapper.find_sessions`. The default is ``None``, in which case the new 
            engine will connect to the first session named in the tuple returned by 
            :meth:`~.MatlabWrapper.find_sessions`.)
        **kwargs :
            Keyword arguments, as documented by the `MATLAB documentation for 
            matlab.engine.connect_matlab <https://uk.mathworks.com/help/matlab/apiref/matlab.engine.connect_matlab.html>`__.

        Returns
        -------
        :class:`matlab.engine.MatlabEngine`
            Python object for communicating with MATLAB.

        Note
        ----
        - Connecting to a shared session requires some initial setup to \
          initialise the shared connection in MATLAB. See \
          `here <https://uk.mathworks.com/help/matlab/ref/matlab.engine.shareengine.html>`_ 
          for more information.
        """  # noqa

        return matlab.engine.connect_matlab(name, *kwargs)

    @staticmethod
    def find_sessions():
        """Discover all shared MATLAB sessions on the local machine.

        This function returns the names of all shared MATLAB sessions.

        Returns
        -------
        :class:`tuple`
            the names of all shared MATLAB sessions running locally.
        """

        return matlab.engine.find_matlab()

    @staticmethod
    def matlab_array(array, array_type=None, size=None, is_complex=None):
        """ Converts an array to a MATLAB array

        Parameters
        ----------
        array: :class:`numpy.ndarray` or :class:`list`
            The array to be converted.
        array_type: :class:`str`, optional
            The type of array to be created. The possible values are ``['single', 'double',
            'logical', 'int8', 'int16', 'int32', 'uint8', 'uint16', 'uint32']``, 
            as per the `MATLAB documentation <https://uk.mathworks.com/help/matlab/matlab_external/matlab-arrays-as-python-variables.html>`__.
            The default value is ``None``, in which case the method will attempt to identify the
            variable type (only ``'logical'``, ``'int32'`` and ``'double'`` types are considered; 
            if the array has mixed types then the method defaults to ``'double'``).
        size: :class:`tuple`, optional
            The size of the array to be created. The default is ``None``. 
        is_complex: :class:`bool`, optional
            Specify whether to create a MATLAB array of complex numbers. The value of this argument 
            is ignored when a logical array is returned. The default is ``False``. 

        Returns
        -------
        :class:`matlab array`
            The converted array

        """  # noqa

        # If it's a numpy array, convert to list; if it's anything
        #  else then throw an error
        if isinstance(array, np.ndarray):
            array = array.tolist()
        elif not isinstance(array, list):
            raise TypeError('Array must be list or numpy.ndarray to convert')

        matlab_funcs = {
            'single': matlab.single, 'double': matlab.double, 'logical': matlab.logical,
            'int8': matlab.int8, 'int16': matlab.int16, 'int32': matlab.int32,
            'uint8': matlab.uint8, 'uint16': matlab.uint16, 'uint32': matlab.uint32}

        if array_type is None:
            # Get a list of all the elements in the array to check the type of
            elems = np.array(array).flatten().tolist()
            if all([type(e) == bool for e in elems]):
                array_type = 'logical'
            elif all([type(e) == int for e in elems]):
                # Defaulting to 32bit because matlab cannot return 64bit ints to python
                array_type = 'int32'
            else:
                # When all else fails, try doubles
                array_type = 'double'
        elif array_type not in matlab_funcs:
            raise ValueError("Invalid value passed for argument array_type. The valid "
                             "values are {}: got '{}'.".format([k for k in matlab_funcs],
                                                               array_type))

        if array_type == 'logical':
            # Logicals cannot be made into an array of complex numbers.
            mat_array = matlab_funcs[array_type](array, size)
        else:
            mat_array = matlab_funcs[array_type](array, size, is_complex)

        return mat_array

    def stop_engine(self):
        """
        End the MATLAB engine
        """

        self.matlab_engine.quit()
        return
