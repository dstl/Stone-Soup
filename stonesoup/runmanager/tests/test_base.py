import multiprocessing as mp
import os
from datetime import datetime

import pathos.multiprocessing

from stonesoup.metricgenerator.base import MetricManager
from stonesoup.runmanager import RunManager
from stonesoup.runmanager.inputmanager import InputManager
from stonesoup.runmanager.runmanagermetrics import RunManagerMetrics
from stonesoup.serialise import YAML
from stonesoup.simulator import GroundTruthSimulator
from stonesoup.tracker import Tracker
from stonesoup.types.groundtruth import GroundTruthPath


def test_conftest_setup(tempdirs, tempfiles):

    # Test that all directories have been generated as expected. If extra test config files or
    # directories are added, they will need to be added to the relevant of the following lists.
    temp_directory = tempdirs['config_dir']

    config_files = {'config_tracker_dets.yaml',
                    'config_tracker_dets_mm.yaml',
                    'config_tracker_gt.yaml',
                    'config_tracker_gt_dets.yaml',
                    'config_tracker_gt_dets_mm.yaml',
                    'config_tracker_gt_mm.yaml',
                    'config_tracker_mm.yaml',
                    'config_tracker_only.yaml',
                    'test_json.json',
                    'test_json_no_run.json',
                    'test_pair.json',
                    'test_pair.yaml'}

    config_subfolders = {'additional_pairs_dir', 'single_file_dir', 'wrong_order_dir'}

    # Check that the temporary directory is generating correctly
    assert type(temp_directory.name) == str

    # Check that the all expected subdirectories from the top level directory are present
    subdirs_found = {f.name for f in os.scandir(temp_directory) if f.is_dir()}
    assert subdirs_found == config_subfolders

    # Check that all expected files in the top level directory are present
    files_found = {f.name for f in os.scandir(temp_directory) if not f.is_dir()}
    assert files_found == config_files

    # Check each of the subdirectories contain the correct files
    subdir = tempdirs['additional_pairs_dir']
    expected_files = {'config_additional_pairs.json', 'config_additional_pairs.yaml'}
    files_found = {f.name for f in os.scandir(subdir) if not f.is_dir()}
    assert files_found == expected_files

    subdir = tempdirs['single_file_dir']
    expected_files = {'config_single_file.yaml'}
    files_found = {f.name for f in os.scandir(subdir) if not f.is_dir()}
    assert files_found == expected_files

    subdir = tempdirs['wrong_order_dir']
    expected_files = {'dummy.json', 'dummy.yaml', 'dummy1.yaml', 'dummy_parameters.json'}
    files_found = {f.name for f in os.scandir(subdir) if not f.is_dir()}
    assert files_found == expected_files


def test_runmanager_init(tempfiles, tempdirs):

    # Values for checking configuration
    nruns = 5
    nprocesses = 1

    rm_args = {"config": tempfiles['config_tracker_gt_mm'],
               "parameters": tempfiles['test_json'],
               "config_dir": tempdirs['config_dir'],
               "nruns": nruns,
               "processes": nprocesses}
    run_manager = RunManager(rm_args=rm_args)

    # Check all provided configuration elements are present
    assert run_manager.config_path == tempfiles['config_tracker_gt_mm']
    assert run_manager.parameters_path == tempfiles['test_json']
    assert run_manager.config_dir == tempdirs['config_dir']
    assert run_manager.nruns == nruns
    assert run_manager.nprocesses == nprocesses

    # Check parameter classes are instantiated correctly
    assert isinstance(run_manager.input_manager, InputManager)
    assert isinstance(run_manager.run_manager_metrics, RunManagerMetrics)

    # Check other set-up values
    assert run_manager.total_trackers == 0
    assert run_manager.current_run == 0
    assert run_manager.total_runs == 0
    assert run_manager.current_trackers == 0


# @pytest.mark.slow
def test_run(tempfiles, tempdirs, tmpdir):
    # This test runs the following configurations:
    # (1) tracker, GT and MM with no json parameter file and no config directory
    # (2) tracker, GT, no MM, with a json parameter file and no config directory
    # (3) no tracker, GT or MM. No json parameter file. Config dir containing both config files
    # (4) no tracker, GT or MM. No json parameter file. Config dir containing yaml config only
    # (5) tracker, GT and MM with a json parameter file and no config directory. No output dir
    configs = [tempfiles['config_tracker_gt_mm'],
               tempfiles['config_tracker_gt'],
               None,
               None,
               tempfiles['config_tracker_gt_mm']]

    parameters = [tempfiles['test_json'],
                  tempfiles['test_json'],
                  None,
                  None,
                  tempfiles['test_json']]

    config_dirs = [None,
                   None,
                   tempdirs['additional_pairs_dir'],
                   tempdirs['single_file_dir'],
                   None]

    nruns_list = [2, 1, 1, 1, 1]
    nprocesses_list = [1, 1, 1, 1, 1]
    output_dirs = [tmpdir, tmpdir, tmpdir, tmpdir, None]
    for config, param, config_dir, nruns, nprocesses, output_dir in zip(configs, parameters,
                                                                        config_dirs, nruns_list,
                                                                        nprocesses_list,
                                                                        output_dirs):
        # Run manager arguments
        rm_args = {"config": config,
                   "parameters": param,
                   "config_dir": config_dir,
                   "nruns": nruns,
                   "processes": nprocesses}

        run_manager = RunManager(rm_args=rm_args)
        if output_dir is not None:
            run_manager.output_dir = output_dir

        run_manager.run()

        if config_dir != tempdirs['single_file_dir']:
            files = [file for file in os.listdir(run_manager.output_dir) if
                     str(run_manager.config_starttime) in str(file)]

            assert len(files) == 1
            out_folder = os.path.join(run_manager.output_dir, files[0])

            sim_folders = os.listdir(out_folder)

            for sim_folder in sim_folders:
                if 'simulation' in str(sim_folder):
                    sim_path = os.path.join(out_folder, sim_folder)

                    if len(os.listdir(sim_path)) > 1:
                        assert 'average.csv' in os.listdir(sim_path)

                    for run_folder in os.listdir(sim_path):
                        if 'run' in str(run_folder):
                            run_path = os.path.join(sim_path, run_folder)

                            # Check that config file has been saved as output
                            assert 'config.yaml' in os.listdir(run_path)

                            # Check that detections, ground truth, and tracks have been saved
                            assert 'detections.csv' in os.listdir(run_path)
                            assert 'groundtruth.csv' in os.listdir(run_path)
                            assert 'tracks.csv' in os.listdir(run_path)
                            # Check metrics are saved if metric manager exists
                            if run_manager.read_config_file()[run_manager.METRIC_MANAGER]:
                                assert 'metrics.csv' in os.listdir(run_path)

        if output_dir is not None:
            assert run_manager.output_dir == output_dir
            assert os.listdir(run_manager.output_dir)
        else:
            assert run_manager.output_dir == "./"


def test_read_json(tempfiles, runmanagers):
    rmc = runmanagers['rmc']
    test_json = tempfiles['test_json']
    test_json_data = rmc.read_json(test_json)

    # Basic checks on json structure
    assert type(test_json_data) is dict

    assert 'configuration' in test_json_data.keys()
    assert 'parameters' in test_json_data.keys()

    assert type(test_json_data['parameters']) == list
    for parameterisation in test_json_data['parameters']:
        assert type(parameterisation) == dict
        assert list(parameterisation.keys()) == ['path', 'type', 'value_min', 'value_max',
                                                 'n_samples']

    # Check output configuration values match the input - values set in
    # conftest.tempfiles.test_json
    assert test_json_data['configuration']['proc_num'] == 1
    assert test_json_data['configuration']['runs_num'] == 4

    # Check output parameter values match the input - values set in conftest.tempfiles.test_json
    parameterisation = test_json_data['parameters'][0]
    assert parameterisation['path'] == 'tracker.initiator.initiator.prior_state.state_vector'
    assert parameterisation['type'] == 'StateVector'
    assert parameterisation['value_min'] == [0, 0, 0, 0]
    assert parameterisation['value_max'] == [1000, 100, 100, 100]
    assert parameterisation['n_samples'] == [1, 0, 0, 0]


def test_set_runs_number(runmanagers, tempfiles, tmpdir):

    # Read in json file that has configuration data
    json_path = tempfiles['test_json']
    json_data = RunManager.read_json(json_path)

    # Read in json file that has no configuration data
    no_run_json_path = tempfiles['test_json_no_run']
    no_run_json_data = RunManager.read_json(no_run_json_path)

    # Import a runmanager #TODO make set_runs_number() a static method?
    rmc = runmanagers['rmc']
    rmc.output_dir = tmpdir

    # Test that input nruns is returned if valid value provided (i.e. 0 < nruns < inf)
    # n_runs should be unchanged
    n = 5
    n_runs = rmc.set_runs_number(nruns=n, json_data=json_data)
    assert n_runs == n

    # Test with nruns = None, json config = True. nruns should be extracted from json file
    n_runs = rmc.set_runs_number(nruns=None, json_data=json_data)
    assert n_runs == json_data['configuration']['runs_num']

    # Test with nruns = 0, json config = True. nruns should be set to default value of 1
    n_runs = rmc.set_runs_number(nruns=0, json_data=json_data)
    assert n_runs == 1

    # Test with nruns = 0, json config = False. nruns should be set to default value of 1
    n_runs = rmc.set_runs_number(nruns=None, json_data=no_run_json_data)
    assert n_runs == 1


def test_set_processes_number(runmanagers, tempfiles, tmpdir):

    # Read in json file that has configuration data
    json_path = tempfiles['test_json']
    json_data = RunManager.read_json(json_path)

    # Read in json file that has no configuration data
    no_run_json_path = tempfiles['test_json_no_run']
    no_run_json_data = RunManager.read_json(no_run_json_path)

    # Import a runmanager #TODO make set_processes_number() a static method?
    rmc = runmanagers['rmc']
    rmc.output_dir = tmpdir

    # Test that input nprocess is returned if valid value provided (i.e. 0 < nprocess < inf)
    # n_runs should be unchanged
    n = 5
    n_proc = rmc.set_processes_number(nprocess=n, json_data=json_data)
    assert n_proc == n

    # Test with nprocess = None, json config = True. n_proc should be extracted from json file
    n_proc = rmc.set_processes_number(nprocess=None, json_data=json_data)
    assert n_proc == json_data['configuration']['proc_num']

    # Test with nprocess = 0, json config = True. n_proc should be set to default value of 1
    n_proc = rmc.set_processes_number(nprocess=0, json_data=json_data)
    assert n_proc == 1

    # Test with nprocess = 0, json config = False. n_proc should be set to default value of 1
    n_proc = rmc.set_processes_number(nprocess=None, json_data=no_run_json_data)
    assert n_proc == 1


def test_prepare_monte_carlo(tmpdir, tempfiles, runmanagers):
    rmc = runmanagers['rmc']
    test_json = tempfiles['test_json']
    rmc.output_dir = tmpdir
    test_json_data = rmc.read_json(test_json)
    test_combo_dict = rmc.prepare_monte_carlo(test_json_data)
    assert len(test_combo_dict) == 24


def test_config_parameter_pairing(tmpdir, runmanagers):
    rmc = runmanagers['rmc']
    rmc_config_dir = runmanagers['rmc_config_dir']
    rmc_config_params_dir = runmanagers['rmc_config_params_dir']

    # Test with config_path and parameters_path
    rmc.output_dir = tmpdir
    test_pairs0 = rmc.config_parameter_pairing()
    assert type(test_pairs0) is list
    assert len(test_pairs0) == 1

    # Test with config_dir
    rmc_config_dir.output_dir = tmpdir
    test_pairs1 = rmc_config_dir.config_parameter_pairing()
    assert type(test_pairs1) is list
    assert len(test_pairs1) == 1

    # Test with config_dir and config_path and parameters_path
    # This also tests params file names: x_parameters.json and x.json both work
    rmc_config_params_dir.output_dir = tmpdir
    test_pairs2 = rmc_config_params_dir.config_parameter_pairing()
    assert type(test_pairs2) is list
    assert len(test_pairs2) == 2


def test_check_ground_truth(tmpdir, runmanagers):
    rmc = runmanagers['rmc']
    rmc.output_dir = tmpdir
    test_gt_no_path = {}
    test_check_gt_no_path = rmc.check_ground_truth(test_gt_no_path)
    assert test_check_gt_no_path == test_gt_no_path


def test_set_trackers(tempfiles, tempdirs, tmpdir):
    # Constant values for all test configurations
    nruns = 5
    nprocesses = 1

    configs = [tempfiles['config_tracker_gt_mm'], tempfiles['config_tracker_gt']]

    test_combo = [{'SingleTargetTracker.initiator.initiator.prior_state.num_particles': 500},
                  {'SingleTargetTracker.initiator.initiator.prior_state.num_particles': 540},
                  {'SingleTargetTracker.initiator.initiator.prior_state.num_particles': 580},
                  {'SingleTargetTracker.initiator.initiator.prior_state.num_particles': 620},
                  {'SingleTargetTracker.initiator.initiator.prior_state.num_particles': 660},
                  {'SingleTargetTracker.initiator.initiator.prior_state.num_particles': 700}]
    empty_combo = []
    combo_no_path = [{'abc': 0}]
    combos = [test_combo, empty_combo, combo_no_path]

    # Iterate through combinations of configs and parameter combinations and check outputs
    for combo in combos:
        for config in configs:
            # Generate arguments for the run manager
            rm_args = {"config": config,
                       "parameters": tempfiles['test_json'],
                       "config_dir": None,
                       "nruns": nruns,
                       "processes": nprocesses}

            # Instantiate run manager
            run_manager = RunManager(rm_args=rm_args)
            run_manager.output_dir = tmpdir

            # Extract data from config file
            with open(config, 'r') as file:
                config_data = YAML(typ='safe').load(file.read())
            file.close()

            if 'tracker' in config_data.keys():
                tracker = config_data['tracker']
            else:
                tracker = None

            if 'groundtruth' in config_data.keys():
                ground_truth = config_data['groundtruth']
            else:
                ground_truth = None

            if 'metric_manager' in config_data.keys():
                metric_manager = config_data['metric_manager']
            else:
                metric_manager = None

            # Run set_trackers() on the configuration elements
            trackers, ground_truths, metric_managers = run_manager.set_trackers(combo,
                                                                                tracker,
                                                                                ground_truth,
                                                                                metric_manager)
            # Basic output check
            assert type(trackers) is list
            assert type(ground_truths) is list
            assert type(metric_managers) is list

            # Check each set of elements is the correct size
            assert len(trackers) == len(combo)
            assert len(ground_truths) == len(combo)
            assert len(metric_managers) == len(combo)

            # Check that each element is a copy of the original
            for tracker_copy, ground_truth_copy, metric_manager_copy in \
                    zip(trackers, ground_truths, metric_managers):
                assert isinstance(tracker_copy, type(tracker))
                assert isinstance(ground_truth_copy, type(ground_truth))
                assert isinstance(metric_manager_copy, type(metric_manager))


def test_set_param(tmpdir, runmanagers):
    rmc = runmanagers['rmc']
    rmc.output_dir = tmpdir
    config_data = rmc.read_config_file()

    tracker = config_data[rmc.TRACKER]

    test_split_path = ['initiator', 'initiator', 'prior_state', 'num_particles']
    test_value = 250

    assert test_split_path[-1] not in dir(tracker.initiator.initiator.prior_state)

    rmc.set_param(test_split_path, tracker, test_value)

    assert test_split_path[-1] in dir(tracker.initiator.initiator.prior_state)
    assert tracker.initiator.initiator.prior_state.num_particles is test_value


def test_set_param_edge_cases(tmpdir, runmanagers):
    rmc = runmanagers['rmc']
    rmc.output_dir = tmpdir
    empty_path = []
    one_path = ['a']
    test_value = 0

    config_data = rmc.read_config_file()

    tracker = config_data[rmc.TRACKER]
    # Empty path
    orig_tracker = tracker
    rmc.set_param(empty_path, tracker, test_value)  # Shouldn't do anything
    assert tracker is orig_tracker

    # Path with one element
    assert 'a' not in dir(tracker)
    rmc.set_param(one_path, tracker, test_value)
    assert 'a' in dir(tracker)
    assert tracker.a is test_value


def test_read_config_file(tempfiles, tmpdir):
    # List of config files to test
    configs = [tempfiles['config_tracker_gt_mm'], tempfiles['config_tracker_gt']]
    nruns = 5
    nprocesses = 1

    for config in configs:
        # Extract data from config file
        with open(config, 'r') as file:
            config_data = YAML(typ='safe').load(file.read())
        file.close()

        if 'tracker' in config_data.keys():
            tracker = config_data['tracker']
        else:
            tracker = None

        if 'groundtruth' in config_data.keys():
            ground_truth = config_data['groundtruth']
        else:
            ground_truth = None

        if 'metric_manager' in config_data.keys():
            metric_manager = config_data['metric_manager']
        else:
            metric_manager = None

        # Generate arguments for the run manager

        rm_args = {"config": config,
                   "parameters": tempfiles['test_json'],
                   "config_dir": None,
                   "nruns": nruns,
                   "processes": nprocesses}

        # Instantiate run manager
        run_manager = RunManager(rm_args=rm_args)
        run_manager.output_dir = tmpdir

        read_config = run_manager.read_config_file()

        # Basic checks on structure
        assert 'tracker' in read_config.keys()
        assert 'ground_truth' in read_config.keys()
        assert 'metric_manager' in read_config.keys()

        # Check that elements read in correctly
        if tracker:
            assert isinstance(read_config['tracker'], Tracker)
        else:
            assert read_config['tracker'] is None
        if ground_truth:
            assert isinstance(read_config['ground_truth'], GroundTruthPath) or \
                   isinstance(read_config['ground_truth'], GroundTruthSimulator)
        else:
            assert read_config['ground_truth'] is None
        if metric_manager:
            assert isinstance(read_config['metric_manager'], MetricManager)
        else:
            assert read_config['metric_manager'] is None

        break


def test_read_config_dir(tmpdir, runmanagers, tempdirs):
    rmc = runmanagers['rmc']
    test_config_dir = tempdirs['config_dir']
    rmc.output_dir = tmpdir
    result = rmc.read_config_dir(test_config_dir)

    # Check output type is a list
    assert type(result) is list

    # Check each item in list is a config file
    for item in result:
        assert '.json' in item or '.yaml' in item


def test_get_filepaths(tmpdir, runmanagers, tempdirs):
    rmc = runmanagers['rmc']
    test_config_dir = tempdirs['config_dir']
    rmc.output_dir = tmpdir
    file_paths = rmc.get_filepaths(test_config_dir)

    # Check each item in list is a config file
    assert type(file_paths) is list

    # Check each item in the list is a file path
    for item in file_paths:
        assert isinstance(item, str)
        assert '.json' in item or '.yaml' in item  # All paths should end with a config file


def test_get_filepaths_empty(tmpdir, runmanagers):
    rmc = runmanagers['rmc']
    rmc.output_dir = tmpdir
    file_path = rmc.get_filepaths('')
    assert len(file_path) == 0


def test_get_config_and_param_lists(tmpdir, tempfiles, runmanagers):
    rmc = runmanagers['rmc']
    rmc_config_dir = runmanagers['rmc_config_dir']
    rmc_config_dir_single_file = runmanagers['rmc_config_dir_single_file']
    rmc.output_dir = tmpdir
    pairs0 = rmc_config_dir.get_config_and_param_lists()
    assert type(pairs0) is list
    assert len(pairs0) == 1
    assert len(pairs0[0]) == 2
    assert os.path.samefile(tempfiles['test_pair_json'],
                            (pairs0[0][1]))
    assert os.path.samefile(tempfiles['test_pair_yaml'], (pairs0[0][0]))

    # Test method can only work with pairs of files
    rmc_config_dir_single_file.output_dir = tmpdir
    pairs1 = rmc_config_dir_single_file.get_config_and_param_lists()
    assert len(pairs1) == 0


def test_get_config_list(tmpdir, runmanagers, tempfiles):
    rmc_config_dir_single_file = runmanagers['rmc_config_dir_single_file']
    rmc_config_dir_single_file.output_dir = tmpdir
    config_list = rmc_config_dir_single_file.get_config_list()
    assert len(config_list) == 1
    assert os.path.samefile(
        tempfiles['config_single_file'],
        config_list[0])


def test_set_components(tmpdir, runmanagers):
    rmc = runmanagers['rmc']
    rmc.output_dir = tmpdir
    config_data = rmc.read_config_file()

    tracker = config_data[rmc.TRACKER]
    ground_truth = config_data[rmc.GROUNDTRUTH]
    metric_manager = config_data[rmc.METRIC_MANAGER]

    assert "tracker" in str(type(tracker))
    assert ground_truth == tracker.detector.groundtruth
    assert "metricgenerator" in str(type(metric_manager))


def test_set_components_no_mm(tmpdir, runmanagers):
    rmc = runmanagers['rmc_nomm']
    rmc.output_dir = tmpdir
    config_data = rmc.read_config_file()

    tracker = config_data.get(rmc.TRACKER)
    ground_truth = config_data.get(rmc.GROUNDTRUTH)
    metric_manager = config_data.get(rmc.METRIC_MANAGER)

    assert "tracker" in str(type(tracker))
    assert ground_truth == tracker.detector.groundtruth
    assert metric_manager is None


def my_testmp_func(x, y):
    return x**y


def test_multiprocess():
    test_process_one = mp.Process(target=my_testmp_func, args=(2, 2))
    test_process_one.start()
    assert test_process_one.is_alive()
    test_process_one.join()
    assert not test_process_one.is_alive()

    test_process_two = mp.Process(target=my_testmp_func, args=(2, 3))
    test_process_three = mp.Process(target=my_testmp_func, args=(3, 2))

    test_process_two.start()
    test_process_three.start()

    assert test_process_two.is_alive()
    assert test_process_three.is_alive()

    test_process_two.join()
    assert not test_process_two.is_alive()
    test_process_three.join()
    assert not test_process_three.is_alive()


def test_multiprocess_pool():
    test_pool = pathos.multiprocessing.ProcessingPool(mp.cpu_count())

    test_mp_result = test_pool.map(my_testmp_func, [2, 2, 3], [2, 3, 2])
    assert test_mp_result == [4, 8, 9]


# TODO resolve logging error at end of tests
def test_single_run(tmpdir, runmanagers, tempfiles):
    rmc = runmanagers['rmc']
    rmc.output_dir = tmpdir
    rmc.config_path = tempfiles['config_tracker_gt_mm']
    rmc.parameters_path = None
    rmc.nruns = None
    rmc.nprocesses = None
    rmc.run()


def test_single_run2(tmpdir, runmanagers, tempfiles):
    rmc = runmanagers['rmc']
    rmc.output_dir = tmpdir
    rmc.config_path = tempfiles['config_tracker_only']
    rmc.parameters_path = None
    rmc.nruns = None
    rmc.nprocesses = None
    rmc.run()


def test_single_run_multiprocess(tmpdir, runmanagers, tempfiles):
    rmc = runmanagers['rmc']
    rmc.output_dir = tmpdir
    rmc.config_path = tempfiles['config_tracker_gt_mm']
    rmc.parameters_path = None
    rmc.nruns = 2
    rmc.nprocesses = 2
    # Catch exception if can't do mp
    try:
        rmc.run()
    except Exception as e:
        print("Couldn't run with multiprocessing: ", e)


# #### Makes tests go on for a while
def test_montecarlo_run(tmpdir, runmanagers, tempfiles):
    rmc = runmanagers['rmc']
    rmc.output_dir = tmpdir
    rmc.config_path = tempfiles['config_tracker_gt_mm']
    rmc.parameters_path = tempfiles['test_json']
    rmc.nruns = None
    rmc.nprocesses = None
    rmc.run()


def test_montecarlo_run_multiprocess(tmpdir, runmanagers, tempfiles):
    rmc = runmanagers['rmc']
    rmc.output_dir = tmpdir
    rmc.config_path = tempfiles['config_tracker_gt_mm']
    rmc.parameters_path = tempfiles['test_json']
    rmc.nruns = 1
    rmc.nprocesses = 2
    # Catch exception if can't do mp
    try:
        rmc.run()
    except Exception as e:
        print("Couldn't run with multiprocessing: ", e)


def test__no_file_run(tmpdir, runmanagers):
    rmc = runmanagers['rmc']
    rmc.output_dir = tmpdir
    rmc.config_path = None
    rmc.parameters_path = None
    rmc.dir = None
    rmc.nruns = None
    rmc.nprocesses = None
    rmc.run()


def test_logging_failed(tmpdir, runmanagers):
    rmc = runmanagers['rmc']
    rmc.output_dir = tmpdir
    rmc.logging_failed_simulation(datetime.now(), "test error message")
