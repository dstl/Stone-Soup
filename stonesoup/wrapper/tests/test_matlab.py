import stonesoup
from stonesoup.wrapper.matlab import MatlabWrapper

###############################
# TODO: Write up more testing #
###############################


def test_wrapper_init():
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
