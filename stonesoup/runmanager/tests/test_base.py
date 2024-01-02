import multiprocessing as mp
import os
from datetime import datetime

import pathos.multiprocessing

from ..base import RunManager

test_config = "stonesoup/runmanager/tests/test_configs/test_config_all.yaml"
test_config_nomm = "stonesoup/runmanager/tests/test_configs/test_config_nomm.yaml"
test_config_trackeronly = "stonesoup/runmanager/tests/test_configs/test_config_trackeronly.yaml"
test_config_dummy = "stonesoup/runmanager/tests/test_configs/dummy.yaml"
test_json = "stonesoup/runmanager/tests/test_configs/dummy_parameters.json"
test_json_no_run = "stonesoup/runmanager/tests/test_configs/dummy_parameters_no_run.json"

test_config_dir = "stonesoup/runmanager/tests/test_configs/"
test_config_dir_single_file = "stonesoup/runmanager/tests/test_configs/test_single_file/"
test_config_dir_additional_pairs = "stonesoup/runmanager/tests/test_configs/test_additional_pairs/"

test_rm_args0 = {"config": test_config,
                 "parameters": test_json,
                 "nruns": 1,
                 "processes": 1}
rmc = RunManager(test_rm_args0)

test_rm_args1 = {"config": test_config_nomm,
                 "parameters": test_json,
                 "nruns": 1,
                 "processes": 1}
rmc_nomm = RunManager(test_rm_args1)

test_rm_args2 = {"config": None,
                 "parameters": None,
                 "config_dir": test_config_dir,
                 "nruns": 1,
                 "processes": 1}
rmc_config_dir = RunManager(test_rm_args2)

test_rm_args3 = {"config": None,
                 "parameters": None,
                 "config_dir": test_config_dir+"test_wrong_order_dir/",
                 "nruns": 1,
                 "processes": 1}
rmc_config_dir_w = RunManager(test_rm_args3)

test_rm_args4 = {"config": test_config_dummy,
                 "parameters": test_json,
                 "config_dir": test_config_dir_additional_pairs,
                 "nruns": 1,
                 "processes": 1}
rmc_config_params_dir = RunManager(test_rm_args4)

test_rm_args5 = {"config": None,
                 "parameters": None,
                 "config_dir": test_config_dir_single_file,
                 "nruns": 1,
                 "processes": 1}
rmc_config_dir_single_file = RunManager(test_rm_args5)


def test_cwd_path():
    assert os.path.isdir('stonesoup/runmanager/tests/test_configs/') is True


def test_read_json():
    test_json_data = rmc.read_json(test_json)
    assert type(test_json_data) is dict


def test_set_runs_number_none(tmpdir):
    rmc.output_dir = tmpdir
    test_json_data = rmc.read_json(test_json)
    test_nruns = rmc.set_runs_number(None, test_json_data)
    assert test_nruns == 4  # nruns set as 4 in dummy_parameters.json


def test_set_runs_number_none_none(tmpdir):
    rmc.output_dir = tmpdir
    test_json_data = rmc.read_json(test_json_no_run)
    test_nruns = rmc.set_runs_number(None, test_json_data)
    assert test_nruns == 1


def test_set_runs_number_one(tmpdir):
    rmc.output_dir = tmpdir
    test_json_data = rmc.read_json(test_json)
    test_nruns = rmc.set_runs_number(1, test_json_data)
    assert test_nruns == 1


def test_set_runs_number_multiple(tmpdir):
    rmc.output_dir = tmpdir
    test_json_data = rmc.read_json(test_json)
    test_nruns = rmc.set_runs_number(2, test_json_data)
    assert test_nruns == 2


def test_set_processes_number_none(tmpdir):
    rmc.output_dir = tmpdir
    test_json_data = rmc.read_json(test_json)
    test_nruns = rmc.set_processes_number(None, test_json_data)
    assert test_nruns == 1  # nprocesses set as 1 in dummy_parameters.json


def test_set_processes_number_none_none(tmpdir):
    rmc.output_dir = tmpdir
    test_json_data = rmc.read_json(test_json_no_run)
    test_nprocess = rmc.set_processes_number(None, test_json_data)
    assert test_nprocess == 1


def test_set_processes_number_one(tmpdir):
    rmc.output_dir = tmpdir
    test_json_data = rmc.read_json(test_json)
    test_nruns = rmc.set_processes_number(1, test_json_data)
    assert test_nruns == 1


def test_set_processes_number_multiple(tmpdir):
    rmc.output_dir = tmpdir
    test_json_data = rmc.read_json(test_json)
    test_nruns = rmc.set_processes_number(2, test_json_data)
    assert test_nruns == 2


def test_prepare_monte_carlo(tmpdir):
    rmc.output_dir = tmpdir
    test_json_data = rmc.read_json(test_json)
    test_combo_dict = rmc.prepare_monte_carlo(test_json_data)
    assert len(test_combo_dict) == 24


def test_config_parameter_pairing(tmpdir):
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


def test_check_ground_truth(tmpdir):
    rmc.output_dir = tmpdir
    test_gt_no_path = {}
    test_check_gt_no_path = rmc.check_ground_truth(test_gt_no_path)
    assert test_check_gt_no_path == test_gt_no_path


def test_set_trackers(tmpdir):
    rmc.output_dir = tmpdir
    test_combo = [{'SingleTargetTracker.initiator.initiator.prior_state.num_particles': 500},
                  {'SingleTargetTracker.initiator.initiator.prior_state.num_particles': 540},
                  {'SingleTargetTracker.initiator.initiator.prior_state.num_particles': 580},
                  {'SingleTargetTracker.initiator.initiator.prior_state.num_particles': 620},
                  {'SingleTargetTracker.initiator.initiator.prior_state.num_particles': 660},
                  {'SingleTargetTracker.initiator.initiator.prior_state.num_particles': 700}]

    with open(test_config, 'r') as file:
        config_data = rmc.read_config_file()
    file.close()

    tracker = config_data[rmc.TRACKER]
    gt = config_data[rmc.GROUNDTRUTH]
    mm = config_data[rmc.METRIC_MANAGER]
    trackers, ground_truths, metric_managers = rmc.set_trackers(test_combo,
                                                                tracker, gt, mm)

    assert type(trackers) is list
    assert type(ground_truths) is list
    assert type(metric_managers) is list

    assert len(trackers) > 0
    assert "tracker" in str(type(trackers[0]))
    assert ground_truths[0] == trackers[0].detector.groundtruth
    assert "metricgenerator" in str(type(metric_managers[0]))


def test_set_trackers_edge_cases(tmpdir):
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


def test_set_param(tmpdir):
    rmc.output_dir = tmpdir
    config_data = rmc.read_config_file()

    tracker = config_data[rmc.TRACKER]

    test_split_path = ['initiator', 'initiator', 'prior_state', 'num_particles']
    test_value = 250

    assert test_split_path[-1] not in dir(tracker.initiator.initiator.prior_state)

    rmc.set_param(test_split_path, tracker, test_value)

    assert test_split_path[-1] in dir(tracker.initiator.initiator.prior_state)
    assert tracker.initiator.initiator.prior_state.num_particles is test_value


def test_set_param_edge_cases(tmpdir):
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


def test_read_config_file(tmpdir):
    rmc.output_dir = tmpdir
    # Config with all tracker, groundtruth, metric manager
    config_data = rmc.read_config_file()

    tracker = config_data[rmc.TRACKER]
    gt = config_data[rmc.GROUNDTRUTH]
    mm = config_data[rmc.METRIC_MANAGER]

    assert "tracker" in str(type(tracker))
    assert gt == tracker.detector.groundtruth
    assert "metricgenerator" in str(type(mm))


def test_read_config_file_nomm(tmpdir):
    rmc.output_dir = tmpdir
    config_data = rmc_nomm.read_config_file()

    tracker = config_data.get(rmc_nomm.TRACKER)
    gt = config_data.get(rmc_nomm.GROUNDTRUTH)
    mm = config_data.get(rmc_nomm.METRIC_MANAGER)
    assert "tracker" in str(type(tracker))
    assert gt == tracker.detector.groundtruth
    assert mm is None


def test_read_config_dir(tmpdir):
    rmc.output_dir = tmpdir
    result = rmc.read_config_dir(test_config_dir)
    assert type(result) is list


def test_read_config_dir_empty(tmpdir):
    rmc.output_dir = tmpdir
    result = rmc.read_config_dir('')
    assert result is None


def test_get_filepaths(tmpdir):
    rmc.output_dir = tmpdir
    file_path = rmc.get_filepaths(test_config_dir)
    assert type(file_path) is list


def test_get_filepaths_empty(tmpdir):
    rmc.output_dir = tmpdir
    file_path = rmc.get_filepaths('')
    assert len(file_path) == 0


def test_get_config_and_param_lists(tmpdir):
    rmc.output_dir = tmpdir
    pairs0 = rmc_config_dir.get_config_and_param_lists()
    assert type(pairs0) is list
    assert len(pairs0) == 1
    assert len(pairs0[0]) == 2
    assert os.path.samefile('stonesoup/runmanager/tests/test_configs/dummy_parameters.json',
                            (pairs0[0][1]))
    assert os.path.samefile('stonesoup/runmanager/tests/test_configs/dummy.yaml', (pairs0[0][0]))

    # Test method can only work with pairs of files
    rmc_config_dir_single_file.output_dir = tmpdir
    pairs1 = rmc_config_dir_single_file.get_config_and_param_lists()
    assert len(pairs1) == 0


def test_get_config_list(tmpdir):
    rmc_config_dir_single_file.output_dir = tmpdir
    config_list = rmc_config_dir_single_file.get_config_list()
    assert len(config_list) == 1
    assert os.path.samefile(
        "stonesoup/runmanager/tests/test_configs/test_single_file/test_config_all.yaml",
        config_list[0])


def test_set_components(tmpdir):
    rmc.output_dir = tmpdir
    config_data = rmc.read_config_file()

    tracker = config_data[rmc.TRACKER]
    ground_truth = config_data[rmc.GROUNDTRUTH]
    metric_manager = config_data[rmc.METRIC_MANAGER]

    assert "tracker" in str(type(tracker))
    assert ground_truth == tracker.detector.groundtruth
    assert "metricgenerator" in str(type(metric_manager))


def test_set_components_no_mm(tmpdir):
    rmc.output_dir = tmpdir
    config_data = rmc_nomm.read_config_file()

    tracker = config_data.get(rmc_nomm.TRACKER)
    ground_truth = config_data.get(rmc_nomm.GROUNDTRUTH)
    metric_manager = config_data.get(rmc_nomm.METRIC_MANAGER)

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
def test_single_run(tmpdir):
    rmc.output_dir = tmpdir
    rmc.config_path = "stonesoup/runmanager/tests/test_configs/dummy.yaml"
    rmc.parameters_path = None
    rmc.nruns = None
    rmc.nprocesses = None
    rmc.run()


def test_single_run_multiprocess(tmpdir):
    rmc.output_dir = tmpdir
    rmc.config_path = "stonesoup/runmanager/tests/test_configs/dummy.yaml"
    rmc.parameters_path = None
    rmc.nruns = 2
    rmc.nprocesses = 2
    # Catch exception if can't do mp
    try:
        rmc.run()
    except Exception as e:
        print("Couldn't run with multiprocessing: ", e)


# #### Makes tests go on for a while
def test_montecarlo_run(tmpdir):
    rmc.output_dir = tmpdir
    rmc.config_path = "stonesoup/runmanager/tests/test_configs/dummy.yaml"
    rmc.parameters_path = "stonesoup/runmanager/tests/test_configs/dummy_parameters.json"
    rmc.nruns = None
    rmc.nprocesses = None
    rmc.run()


def test_montecarlo_run_multiprocess(tmpdir):
    rmc.output_dir = tmpdir
    rmc.config_path = "stonesoup/runmanager/tests/test_configs/dummy.yaml"
    rmc.parameters_path = "stonesoup/runmanager/tests/test_configs/dummy_parameters.json"
    rmc.nruns = 1
    rmc.nprocesses = 2
    # Catch exception if can't do mp
    try:
        rmc.run()
    except Exception as e:
        print("Couldn't run with multiprocessing: ", e)


def test__no_file_run(tmpdir):
    rmc.output_dir = tmpdir
    rmc.config_path = None
    rmc.parameters_path = None
    rmc.dir = None
    rmc.nruns = None
    rmc.nprocesses = None
    rmc.run()


def test_logging_failed(tmpdir):
    rmc.output_dir = tmpdir
    rmc.logging_failed_simulation(datetime.now(), "test error message")
