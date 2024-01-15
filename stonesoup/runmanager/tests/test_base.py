import multiprocessing as mp
import os
from datetime import datetime
import pathos.multiprocessing

import stonesoup.metricgenerator
from stonesoup.serialise import YAML


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


def test_read_json(tempfiles, runmanagers):
    rmc = runmanagers['rmc']
    test_json = tempfiles['test_json']
    test_json_data = rmc.read_json(test_json)
    assert type(test_json_data) is dict


def test_set_runs_number_none(tmpdir, tempfiles, runmanagers):
    rmc = runmanagers['rmc']
    test_json = tempfiles['test_json']
    rmc.output_dir = tmpdir
    test_json_data = rmc.read_json(test_json)
    test_nruns = rmc.set_runs_number(None, test_json_data)
    assert test_nruns == 4  # nruns set as 4 in dummy_parameters.json


def test_set_runs_number_none_none(tmpdir, tempfiles, runmanagers):
    rmc = runmanagers['rmc']
    test_json_no_run = tempfiles['test_json_no_run']
    rmc.output_dir = tmpdir
    test_json_data = rmc.read_json(test_json_no_run)
    test_nruns = rmc.set_runs_number(None, test_json_data)
    assert test_nruns == 1


def test_set_runs_number_one(tmpdir, tempfiles, runmanagers):
    rmc = runmanagers['rmc']
    test_json = tempfiles['test_json']
    rmc.output_dir = tmpdir
    test_json_data = rmc.read_json(test_json)
    test_nruns = rmc.set_runs_number(1, test_json_data)
    assert test_nruns == 1


def test_set_runs_number_multiple(tmpdir, tempfiles, runmanagers):
    rmc = runmanagers['rmc']
    test_json = tempfiles['test_json']
    rmc.output_dir = tmpdir
    test_json_data = rmc.read_json(test_json)
    test_nruns = rmc.set_runs_number(2, test_json_data)
    assert test_nruns == 2


def test_set_processes_number_none(tmpdir, tempfiles, runmanagers):
    rmc = runmanagers['rmc']
    test_json = tempfiles['test_json']
    rmc.output_dir = tmpdir
    test_json_data = rmc.read_json(test_json)
    test_nruns = rmc.set_processes_number(None, test_json_data)
    assert test_nruns == 1  # nprocesses set as 1 in dummy_parameters.json


def test_set_processes_number_none_none(tmpdir, tempfiles, runmanagers):
    rmc = runmanagers['rmc']
    test_json_no_run = tempfiles['test_json_no_run']
    rmc.output_dir = tmpdir
    test_json_data = rmc.read_json(test_json_no_run)
    test_nprocess = rmc.set_processes_number(None, test_json_data)
    assert test_nprocess == 1


def test_set_processes_number_one(tmpdir, tempfiles, runmanagers):
    rmc = runmanagers['rmc']
    test_json = tempfiles['test_json']
    rmc.output_dir = tmpdir
    test_json_data = rmc.read_json(test_json)
    test_nruns = rmc.set_processes_number(1, test_json_data)
    assert test_nruns == 1


def test_set_processes_number_multiple(tmpdir, tempfiles, runmanagers):
    rmc = runmanagers['rmc']
    test_json = tempfiles['test_json']
    rmc.output_dir = tmpdir
    test_json_data = rmc.read_json(test_json)
    test_nruns = rmc.set_processes_number(2, test_json_data)
    assert test_nruns == 2


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


def test_set_trackers(tmpdir, runmanagers, tempfiles):
    rmc = runmanagers['rmc']
    test_config = tempfiles['config_tracker_gt_mm']
    rmc.output_dir = tmpdir
    test_combo = [{'SingleTargetTracker.initiator.initiator.prior_state.num_particles': 500},
                  {'SingleTargetTracker.initiator.initiator.prior_state.num_particles': 540},
                  {'SingleTargetTracker.initiator.initiator.prior_state.num_particles': 580},
                  {'SingleTargetTracker.initiator.initiator.prior_state.num_particles': 620},
                  {'SingleTargetTracker.initiator.initiator.prior_state.num_particles': 660},
                  {'SingleTargetTracker.initiator.initiator.prior_state.num_particles': 700}]

    with open(test_config, 'r') as file:
        config_data = YAML(typ='safe').load(file.read())
    file.close()

    tracker = config_data['tracker']
    gt = config_data['groundtruth']
    mm = config_data['metric_manager']

    trackers, ground_truths, metric_managers = rmc.set_trackers(test_combo,
                                                                tracker, gt, mm)

    assert type(trackers) is list
    assert type(ground_truths) is list
    assert type(metric_managers) is list

    assert len(trackers) > 0
    assert "tracker" in str(type(trackers[0]))
    assert ground_truths[0] == trackers[0].detector.groundtruth
    assert isinstance(metric_managers[0], stonesoup.metricgenerator.manager.SimpleManager)


def test_set_trackers_edge_cases(tmpdir, runmanagers, tempfiles):
    rmc = runmanagers['rmc']
    test_config = tempfiles['config_tracker_gt_mm']
    rmc.output_dir = tmpdir
    empty_combo = []
    combo_no_path = [{'abc': 0}]

    with open(test_config, 'r') as file:
        config_data = rmc.read_config_file()
    file.close()

    tracker = config_data[rmc.TRACKER]
    gt = config_data[rmc.GROUNDTRUTH]
    mm = config_data[rmc.METRIC_MANAGER]
    # Empty combo dict
    trackers, ground_truths, metric_managers = rmc.set_trackers(empty_combo,
                                                                tracker, gt, mm)

    assert type(trackers) is list
    assert type(ground_truths) is list
    assert type(metric_managers) is list
    assert len(trackers) == 0
    assert len(ground_truths) == 0
    assert len(metric_managers) == 0

    # No path combo dict
    trackers, ground_truths, metric_managers = rmc.set_trackers(combo_no_path,
                                                                tracker, gt, mm)

    assert type(trackers) is list
    assert type(ground_truths) is list
    assert type(metric_managers) is list
    assert len(trackers) == 1
    assert len(ground_truths) == 1
    assert len(metric_managers) == 1


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


def test_read_config_file(tmpdir, runmanagers):
    rmc = runmanagers['rmc']
    rmc.output_dir = tmpdir
    # Config with all tracker, groundtruth, metric manager
    config_data = rmc.read_config_file()

    tracker = config_data[rmc.TRACKER]
    gt = config_data[rmc.GROUNDTRUTH]
    mm = config_data[rmc.METRIC_MANAGER]

    assert "tracker" in str(type(tracker))
    assert gt == tracker.detector.groundtruth
    assert "metricgenerator" in str(type(mm))


def test_read_config_file_nomm(tmpdir, runmanagers):
    rmc = runmanagers['rmc']
    rmc_nomm = runmanagers['rmc_nomm']
    rmc.output_dir = tmpdir
    config_data = rmc_nomm.read_config_file()

    tracker = config_data.get(rmc_nomm.TRACKER)
    gt = config_data.get(rmc_nomm.GROUNDTRUTH)
    mm = config_data.get(rmc_nomm.METRIC_MANAGER)
    assert "tracker" in str(type(tracker))
    assert gt == tracker.detector.groundtruth
    assert mm is None


def test_read_config_dir(tmpdir, runmanagers, tempdirs):
    rmc = runmanagers['rmc']
    test_config_dir = tempdirs['config_dir']
    rmc.output_dir = tmpdir
    result = rmc.read_config_dir(test_config_dir)
    assert type(result) is list


def test_read_config_dir_empty(tmpdir, runmanagers):
    rmc = runmanagers['rmc']
    rmc.output_dir = tmpdir
    result = rmc.read_config_dir('')
    assert result is None


def test_get_filepaths(tmpdir, runmanagers, tempdirs):
    rmc = runmanagers['rmc']
    test_config_dir = tempdirs['config_dir']
    rmc.output_dir = tmpdir
    file_path = rmc.get_filepaths(test_config_dir)
    assert type(file_path) is list


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
