import pytest

try:
    from stonesoup.wrapper.matlab import MatlabWrapper
except ImportError:
    # Catch optional dependencies import error
    pytest.skip("Skipping due to missing optional dependencies. Usage of the MatlabWrapper class "
                "requires that MATLAB and the MATLAB Engine API for Python are installed. More "
                "information can be found here: "
                "https://uk.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-"
                "for-python.html",
                allow_module_level=True)
import stonesoup
import numpy as np
import matlab


def test_matlab_wrapper_init():
    """Test that the Matlab wrapper gets initialised correctly

    Note
    ====
    Takes a while to run as starting new engines adds significant
    overheads and we are starting 3 engines here.
    """

    # 1) No init args
    wrapper = MatlabWrapper()
    assert(wrapper.dir_path == stonesoup.__file__.strip('__init__.py'))

    # 2) ONLY dir_path
    dir_path = "./"
    wrapper = MatlabWrapper(dir_path)
    assert(wrapper.dir_path == dir_path)

    # 3) Both dir_path and external engine
    engine = MatlabWrapper.start_engine()
    wrapper = MatlabWrapper(dir_path, engine)
    assert(wrapper.dir_path == dir_path)
    assert(wrapper.matlab_engine == engine)


def test_matlab_wrapper_array():

    arr = np.array([0, 1])

    matlab_types = {
        'single': matlab.single, 'double': matlab.double, 'logical': matlab.logical,
        'int8': matlab.int8, 'int16': matlab.int16, 'int32': matlab.int32,
        'uint8': matlab.uint8, 'uint16': matlab.uint16, 'uint32': matlab.uint32}

    wrapper = MatlabWrapper()

    for arr_type in matlab_types:
        mat_arr = wrapper.matlab_array(arr, array_type=arr_type)
        assert(isinstance(mat_arr, matlab_types.get(arr_type)))

    with pytest.raises(ValueError):
        mat_arr = wrapper.matlab_array(arr, array_type='uint64')
