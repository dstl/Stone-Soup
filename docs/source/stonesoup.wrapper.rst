Wrappers
========

Objects that include calls to code in different languages should inherit from
the relevant wrapper along with the standard inheritance. The wrapper objects
include functions for managing the bridge between languages and converting
between data types.

.. automodule:: stonesoup.wrapper
    :no-members:

.. automodule:: stonesoup.wrapper.base
    :show-inheritance:

MATLAB wrapper
==============

The connection between MATLAB and python is handled by the API that comes with
modern versions of MATLAB. For more information see
https://uk.mathworks.com/help/matlab/matlab-engine-for-python.html

The current version of the API works with python versions 2.7, 3.5 and 3.6 and
both the MATLAB and python installs need to be the same bit version (32 or 64).
MATLAB’s support for this API appears to be improving significantly with each
version. This interface was written for MATLAB version R2017a and Python 3.5.
Version R2018b appears to have fixed a couple of the issues such as being
unable to handle 64bit integers. However there may have also been some
changes to how the engine is called so using this with other versions
of MATLAB may require some changes. Also note that the API does not
support python 3.7, no testing has been done on whether it still
works.

To call MATLAB functions in Stone Soup as part of an object the object
needs to inherit from the MatlabWrapper object. When the object is created
an instance of the MATLAB engine is started and remains active. To call a
function in an object that inherits from MatlabWrapper use
    output_1, … , output_n = self.matlab_engine.<function name>(argument1, agument2, … , nargout = n)
Where nargout defines the number of expected arguments to be returned
(can be zero). Each of the arguments to the function needs to be in the form
of a matlab array which can be created from a list or numpy array using
MatlabWrapper.matlab_array(x). The type of the output array depends on
the type of the input array (e.g. and array of integers is converted
to a matlab integer array).

Both the file name and function name within the file need to be the same.
For example to call a function test_function would require the function to
be in a file called test_function.m.

See stonesoup/examples/wrappers/matlab/predictor for an example


.. automodule:: stonesoup.wrapper.matlab
    :show-inheritance:
