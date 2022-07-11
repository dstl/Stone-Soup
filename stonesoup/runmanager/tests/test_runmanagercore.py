import os
from datetime import datetime
import multiprocessing as mp
from pathos.multiprocessing import ProcessingPool as Pool
from ..runmanagercore import RunManagerCore

# Run from stonesoup working directory
# def setup_module():
#     while os.getcwd().split('\\')[-1] != 'stonesoup':
#         os.chdir(os.path.dirname(os.getcwd()))

# test_config = "stonesoup\\runmanager\\tests\\test_configs\\test_config_all.yaml"
# test_config_nomm = "stonesoup\\runmanager\\tests\\test_configs\\test_config_nomm.yaml"
# test_config_nogt = "stonesoup\\runmanager\\tests\\test_configs\\test_config_nogt.yaml"
# test_config_trackeronly =
# "stonesoup\\runmanager\\tests\\test_configs\\test_config_trackeronly.yaml"
# test_json = "stonesoup\\runmanager\\tests\\test_configs\\dummy_parameters.json"


test_config = "stonesoup/runmanager/tests/test_configs/test_config_all.yaml"
test_config_nomm = "stonesoup/runmanager/tests/test_configs/test_config_nomm.yaml"
test_config_nogt = "stonesoup/runmanager/tests/test_configs/test_config_nogt.yaml"
test_config_trackeronly = "stonesoup/runmanager/tests/test_configs/test_config_trackeronly.yaml"
test_json = "stonesoup/runmanager/tests/test_configs/dummy_parameters.json"
test_json_no_run = "stonesoup/runmanager/tests/test_configs/dummy_parameters_no_run.json"

test_config_dir = "stonesoup/runmanager/tests/test_configs/"

rmc = RunManagerCore(test_config, test_json, False, False, test_config_dir, 1, 1)


def test_cwd_path():
    assert os.path.isdir('stonesoup/runmanager/tests/test_configs') is True


def test_read_json():

    test_json_data = rmc.read_json(test_json)
    assert type(test_json_data) is dict


def test_set_runs_number_none():
    test_json_data = rmc.read_json(test_json)
    test_nruns = rmc.set_runs_number(None, test_json_data)
    assert test_nruns == 4  # nruns set as 4 in dummy_parameters.json


def test_set_runs_number_none_none():
    test_json_data = rmc.read_json(test_json_no_run)
    test_nruns = rmc.set_runs_number(None, test_json_data)
    assert test_nruns == 1


def test_set_runs_number_one():
    test_json_data = rmc.read_json(test_json)
    test_nruns = rmc.set_runs_number(1, test_json_data)
    assert test_nruns == 1


def test_set_runs_number_multiple():
    test_json_data = rmc.read_json(test_json)
    test_nruns = rmc.set_runs_number(2, test_json_data)
    assert test_nruns == 2


def test_set_processes_number_none():
    test_json_data = rmc.read_json(test_json)
    test_nruns = rmc.set_processes_number(None, test_json_data)
    assert test_nruns == 1  # nprocesses set as 1 in dummy_parameters.json


def test_set_processes_number_none_none():
    test_json_data = rmc.read_json(test_json_no_run)
    test_nprocess = rmc.set_processes_number(None, test_json_data)
    assert test_nprocess == 1


def test_set_processes_number_one():
    test_json_data = rmc.read_json(test_json)
    test_nruns = rmc.set_processes_number(1, test_json_data)
    assert test_nruns == 1


def test_set_processes_number_multiple():
    test_json_data = rmc.read_json(test_json)
    test_nruns = rmc.set_processes_number(2, test_json_data)
    assert test_nruns == 2


def test_prepare_monte_carlo():
    test_json_data = rmc.read_json(test_json)
    test_combo_dict = rmc.prepare_monte_carlo(test_json_data)
    assert len(test_combo_dict) == 24


def test_config_parameter_pairing():
    test_pairs = rmc.config_parameter_pairing()

    assert type(test_pairs) is list
    assert len(test_pairs) == 1


def test_check_ground_truth():
    test_gt_no_path = {}
    test_check_gt_no_path = rmc.check_ground_truth(test_gt_no_path)
    assert test_check_gt_no_path == test_gt_no_path


def test_set_trackers():

    test_combo = [{'SingleTargetTracker.initiator.initiator.prior_state.num_particles': 500},
                  {'SingleTargetTracker.initiator.initiator.prior_state.num_particles': 540},
                  {'SingleTargetTracker.initiator.initiator.prior_state.num_particles': 580},
                  {'SingleTargetTracker.initiator.initiator.prior_state.num_particles': 620},
                  {'SingleTargetTracker.initiator.initiator.prior_state.num_particles': 660},
                  {'SingleTargetTracker.initiator.initiator.prior_state.num_particles': 700}]

    with open(test_config, 'r') as file:
        config_data = rmc.read_config_file(file)
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


def test_set_trackers_edge_cases():

    empty_combo = []
    combo_no_path = [{'abc': 0}]

    with open(test_config, 'r') as file:
        config_data = rmc.read_config_file(file)
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


def test_set_param():

    with open(test_config, 'r') as file:
        config_data = rmc.read_config_file(file)

    tracker = config_data[rmc.TRACKER]

    test_split_path = ['initiator', 'initiator', 'prior_state', 'num_particles']
    test_value = 250

    assert test_split_path[-1] not in dir(tracker.initiator.initiator.prior_state)

    rmc.set_param(test_split_path, tracker, test_value)

    assert test_split_path[-1] in dir(tracker.initiator.initiator.prior_state)
    assert tracker.initiator.initiator.prior_state.num_particles is test_value


def test_set_param_edge_cases():
    empty_path = []
    one_path = ['a']
    test_value = 0

    with open(test_config, 'r') as file:
        config_data = rmc.read_config_file(file)

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


def test_read_config_file():

    # Config with all tracker, grountruth, metric manager
    with open(test_config, 'r') as file:
        config_data = rmc.read_config_file(file)

    tracker = config_data[rmc.TRACKER]
    gt = config_data[rmc.GROUNDTRUTH]
    mm = config_data[rmc.METRIC_MANAGER]

    assert "tracker" in str(type(tracker))
    assert gt == tracker.detector.groundtruth
    assert "metricgenerator" in str(type(mm))
    file.close()


def test_read_config_file_nomm():

    # Config with tracker and groundtruth but no metric manager
    with open(test_config_nomm, 'r') as file:
        config_data = rmc.read_config_file(file)

    tracker = config_data[rmc.TRACKER]
    gt = config_data[rmc.GROUNDTRUTH]
    mm = config_data[rmc.METRIC_MANAGER]
    assert "tracker" in str(type(tracker))
    assert gt == tracker.detector.groundtruth
    assert mm is None
    file.close()


def test_read_config_dir():
    result = rmc.read_config_dir(test_config_dir)
    assert type(result) is list


def test_read_config_dir_empty():
    result = rmc.read_config_dir('')
    assert result is None


def test_get_filepaths():
    file_path = rmc.get_filepaths(test_config_dir)
    assert type(file_path) is list


def test_get_filepaths_empty():
    file_path = rmc.get_filepaths('')
    assert len(file_path) == 0


# TODO this test
def test_order_pairs():
    pass


def test_get_config_and_param_lists():
    files = rmc.get_filepaths(test_config_dir)
    pairs = rmc.get_config_and_param_lists(files)
    assert type(pairs) is list
    assert len(pairs) == 1
    assert len(pairs[0]) == 2
    assert 'stonesoup/runmanager/tests/test_configs/dummy_parameters.json' == (pairs[0][1])
    assert 'stonesoup/runmanager/tests/test_configs/dummy.yaml' in (pairs[0][0])


def test_get_config_and_param_lists_wrong_order():
    files = rmc.get_filepaths('stonesoup/runmanager/tests/test_configs/test_wrong_order_dir/')
    pairs = rmc.get_config_and_param_lists(files)
    assert type(pairs) is list
    assert len(pairs) == 2
    assert len(pairs[0]) == 2
    assert len(pairs[1]) == 2
    assert ('stonesoup/runmanager/tests/test_configs' +
            '/test_wrong_order_dir/dummy.json') == (pairs[0][1])
    assert ('stonesoup/runmanager/tests/test_configs' +
            '/test_wrong_order_dir/dummy.yaml') == (pairs[0][0])
    assert ('stonesoup/runmanager/tests/test_configs' +
            '/test_wrong_order_dir/dummy1.json') == (pairs[1][1])
    assert ('stonesoup/runmanager/tests/test_configs' +
            '/test_wrong_order_dir/dummy1.yaml') == (pairs[1][0])


def test_set_components_empty():
    config_data = rmc.set_components('')

    tracker = config_data[rmc.TRACKER]
    ground_truth = config_data[rmc.GROUNDTRUTH]
    metric_manager = config_data[rmc.METRIC_MANAGER]

    assert tracker is None
    assert ground_truth is None
    assert metric_manager is None


def test_set_components():
    config_data = rmc.set_components(test_config)

    tracker = config_data[rmc.TRACKER]
    ground_truth = config_data[rmc.GROUNDTRUTH]
    metric_manager = config_data[rmc.METRIC_MANAGER]

    assert "tracker" in str(type(tracker))
    assert ground_truth == tracker.detector.groundtruth
    assert "metricgenerator" in str(type(metric_manager))


def test_set_components_no_mm():
    config_data = rmc.set_components(test_config_nomm)

    tracker = config_data[rmc.TRACKER]
    ground_truth = config_data[rmc.GROUNDTRUTH]
    metric_manager = config_data[rmc.METRIC_MANAGER]

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
    test_pool = Pool(mp.cpu_count())

    test_mp_result = test_pool.map(my_testmp_func, [2, 2, 3], [2, 3, 2])
    assert test_mp_result == [4, 8, 9]


# TODO resolve logging error at end of tests
def test_single_run():
    rmc.config_path = "stonesoup/runmanager/tests/test_configs/dummy.yaml"
    rmc.parameters_path = None
    rmc.nruns = None
    rmc.nprocesses = None
    rmc.run()


def test_single_run_multiprocess():
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
def test_montecarlo_run():
    rmc.config_path = "stonesoup/runmanager/tests/test_configs/dummy.yaml"
    rmc.parameters_path = "stonesoup/runmanager/tests/test_configs/dummy_parameters.json"
    rmc.nruns = None
    rmc.nprocesses = None
    rmc.run()


def test_montecarlo_run_multiprocess():
    rmc.config_path = "stonesoup/runmanager/tests/test_configs/dummy.yaml"
    rmc.parameters_path = "stonesoup/runmanager/tests/test_configs/dummy_parameters.json"
    rmc.nruns = 1
    rmc.nprocesses = 2
    # Catch exception if can't do mp
    try:
        rmc.run()
    except Exception as e:
        print("Couldn't run with multiprocessing: ", e)


def test__no_file_run():
    rmc.config_path = None
    rmc.parameters_path = None
    rmc.dir = None
    rmc.nruns = None
    rmc.nprocesses = None
    rmc.run()


def test_logging_failed():
    rmc.logging_failed_simulation(datetime.now(), "test error message")


# TODO: this test
def test_setup_logger():
    pass
