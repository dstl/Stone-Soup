# -*- coding: utf-8 -*-
import os
import pytest
import tempfile
import copy
import shutil
from stonesoup.deleter.error import CovarianceBasedDeleter
from stonesoup.deleter.time import UpdateTimeStepsDeleter
from stonesoup.runmanager.manager import SimpleRunManager, output_data
from stonesoup.reader.aishub import JSON_AISDetectionReader
from stonesoup.runmanager.manager import output_folder


def test_run_manager_base_case(base_tracker):
    # runs tests on some basic Run Manager configurations

    # --------------------------------------------
    #   create a standard Tracker
    # --------------------------------------------
    local_tracker = base_tracker.get_base_tracker()

    # --------------------------------------------
    #   run the standard Tracker
    # --------------------------------------------
    local_run_manager = SimpleRunManager(run_checks=True,
                                         illegal_attribute_values=[None])
    local_run_manager.load_config(object=local_tracker)

    local_run_manager.run_experiment(save_detections_tracks=True)
    local_run_manager.output_experiment_results()

    # -------------------------------------------------------------------
    # check that all of the experiment results were generated correctly
    # -------------------------------------------------------------------
    metrics_results = local_run_manager.\
        experiment_results['Experiment00001']['results'].values()
    assert any(x.title == 'Number of targets' for x in metrics_results)
    assert any(x.title == 'Number of tracks' for x in metrics_results)
    assert any(x.title == 'Track-to-target ratio' for x in metrics_results)
    assert any(x.title == 'SIAP A' for x in metrics_results)
    assert any(x.title == 'SIAP C' for x in metrics_results)
    assert any(x.title == 'SIAP S' for x in metrics_results)
    assert any(x.title == 'SIAP LS' for x in metrics_results)
    assert any(x.title == 'SIAP LT' for x in metrics_results)
    assert any(x.title == 'OSPA distances' for x in metrics_results)

    # -------------------------------------------------------------------
    # verify that metrics have been written to output file
    # -------------------------------------------------------------------
    assert os.path.exists(local_run_manager.experiment_results_filename)

    # -------------------------------------------------------------------
    # test reading 'config' from an output file
    # -------------------------------------------------------------------
    file = tempfile.NamedTemporaryFile(prefix="runmanager_test_", delete=False)
    file.close()
    output_data(file.name, local_tracker)

    local_run_manager = SimpleRunManager(run_checks=True,
                                         illegal_attribute_values=[None])
    local_run_manager.load_config(filename=file.name)

    local_run_manager.run_experiment()
    local_run_manager.output_experiment_results()

    metrics_results = local_run_manager.\
        experiment_results['Experiment00001']['results'].values()
    assert any(x.title == 'Number of targets' for x in metrics_results)
    assert any(x.title == 'Number of tracks' for x in metrics_results)
    assert any(x.title == 'Track-to-target ratio' for x in metrics_results)
    assert any(x.title == 'SIAP A' for x in metrics_results)
    assert any(x.title == 'SIAP C' for x in metrics_results)
    assert any(x.title == 'SIAP S' for x in metrics_results)
    assert any(x.title == 'SIAP LS' for x in metrics_results)
    assert any(x.title == 'SIAP LT' for x in metrics_results)
    assert any(x.title == 'OSPA distances' for x in metrics_results)
    assert os.path.exists(local_run_manager.experiment_results_filename)

    # ----------------------------------------------------
    #   test case where there are no metrics to output
    # ----------------------------------------------------
    local_run_manager.experiment_results = dict()
    local_run_manager.output_experiment_results()

    # ------------------------------------------------------------------
    #   test case where we do not check the experiment configuration
    #   before running it
    # ------------------------------------------------------------------
    local_run_manager = SimpleRunManager(run_checks=False,
                                         illegal_attribute_values=[None])
    local_run_manager.load_config(object=local_tracker)
    local_run_manager.run_experiment()


def test_run_manager_monte_carlo(base_tracker):
    # tests the Run Manager with a Monte Carlo experiment

    # ------------------------------------------------------------------------
    #   create a standard Tracker and modify it to be a Monte Carlo experiment
    # ------------------------------------------------------------------------
    local_tracker = base_tracker.get_base_tracker()

    local_tracker['conditions'] = {"monte_carlo": 3, "output_data": True}

    # --------------------------------------------
    #   run the Tracker
    # --------------------------------------------
    local_run_manager = SimpleRunManager(run_checks=True,
                                         illegal_attribute_values=[None])
    local_run_manager.load_config(object=local_tracker)

    local_run_manager.run_experiment()
    local_run_manager.output_experiment_results()

    # -------------------------------------------------------------------
    # check that all of the experiment results were generated correctly
    # -------------------------------------------------------------------
    for experiment_name in ['Experiment00001', 'Experiment00002',
                            'Experiment00003']:
        metrics_results = local_run_manager.\
            experiment_results[experiment_name]['results'].values()
        assert any(x.title == 'Number of targets' for x in metrics_results)
        assert any(x.title == 'Number of tracks' for x in metrics_results)
        assert any(x.title == 'Track-to-target ratio' for x in metrics_results)
        assert any(x.title == 'SIAP A' for x in metrics_results)
        assert any(x.title == 'SIAP C' for x in metrics_results)
        assert any(x.title == 'SIAP S' for x in metrics_results)
        assert any(x.title == 'SIAP LS' for x in metrics_results)
        assert any(x.title == 'SIAP LT' for x in metrics_results)
        assert any(x.title == 'OSPA distances' for x in metrics_results)


def test_run_manager_tracker_variants(base_tracker):
    # tests a Run Manager that is given several Tracker variants

    # ------------------------------------------------------------------------
    #   create a standard Tracker and modify it to have several variants
    # ------------------------------------------------------------------------
    local_tracker = base_tracker.get_base_tracker()

    deleter01 = CovarianceBasedDeleter(covar_trace_thresh=1E3)
    deleter02 = UpdateTimeStepsDeleter(5)

    local_tracker['tracker01'].detector.clutter_rate = iter([0.3, 0.2])

    local_tracker['components'] = \
        {"tracker01.deleter": iter([deleter01, deleter02]),
         "tracker01.detector.clutter_rate": 0.1,
         "tracker02.half_deleter": deleter01}

    # ------------------------------------------------------------------------
    #   run the Tracker - multiple experiments due to variants
    # ------------------------------------------------------------------------
    local_run_manager = SimpleRunManager(run_checks=True,
                                         illegal_attribute_values=[None])
    local_run_manager.load_config(object=local_tracker)

    local_run_manager.run_experiment()
    local_run_manager.output_experiment_results()

    # -------------------------------------------------------------------
    # check that all of the experiment results were generated correctly
    # -------------------------------------------------------------------
    for experiment_name in ['Experiment00001', 'Experiment00002',
                            'Experiment00003']:
        metrics_results = local_run_manager.\
            experiment_results[experiment_name]['results'].values()
        assert any(x.title == 'Number of targets' for x in metrics_results)
        assert any(x.title == 'Number of tracks' for x in metrics_results)
        assert any(x.title == 'Track-to-target ratio' for x in metrics_results)
        assert any(x.title == 'SIAP A' for x in metrics_results)
        assert any(x.title == 'SIAP C' for x in metrics_results)
        assert any(x.title == 'SIAP S' for x in metrics_results)
        assert any(x.title == 'SIAP LS' for x in metrics_results)
        assert any(x.title == 'SIAP LT' for x in metrics_results)
        assert any(x.title == 'OSPA distances' for x in metrics_results)


def test_run_manager_import_data(base_tracker):
    # tests a Run Manager that imports data from a file for 'tracker.detector'

    # -------------------------------------------------------------------
    #   create a "data file" for detections to be read from
    # -------------------------------------------------------------------
    test_file_text = \
        "[{\"ERROR\": \"false\"}," + \
        "[{\"NAME\": \"COSCO VIETNAM\", \"MMSI\": 477266900," + \
        "\"LONGITUDE\": 2419589, \"TIME\": \"1516320059\"," + \
        "\"LATITUDE\": 31185128}," + \
        "{\"NAME\": \"COSCO VIETNAM\", \"MMSI\": 477266900," + \
        "\"LONGITUDE\": 2419403, \"TIME\": \"1516320603\"," + \
        "\"LATITUDE\": 31185103}," + \
        "{\"NAME\": \"COSCO VIETNAM\", \"MMSI\": 477266900," + \
        "\"LONGITUDE\": 2419354, \"TIME\": \"1516320962\"," + \
        "\"LATITUDE\": 31185078}," + \
        "{\"NAME\": \"COSCO VIETNAM\", \"MMSI\": 477266900," + \
        "\"LONGITUDE\": 2419531, \"TIME\": \"1516321680\"," + \
        "\"LATITUDE\": 31185027}]]"

    file = tempfile.NamedTemporaryFile(prefix="runmanager_test_", delete=False)
    file.close()
    f = open(file.name, 'w')
    f.write(test_file_text)
    f.close()

    # ------------------------------------------------------------------------
    #   create a tracker and set 'detections' to a file reader
    # ------------------------------------------------------------------------
    local_tracker = base_tracker.get_base_tracker()

    local_tracker['tracker01'].detector = \
        JSON_AISDetectionReader(path=file.name)

    # ---------------------------------
    #   run the Tracker
    # ---------------------------------
    local_run_manager = SimpleRunManager(run_checks=True,
                                         illegal_attribute_values=[None])
    local_run_manager.load_config(object=local_tracker)

    local_run_manager.run_experiment()

    del local_tracker['tracker01'].detector

    # ------------------------------------------------------------------------
    #   sanity check the experiment results
    # ------------------------------------------------------------------------
    metrics_results = local_run_manager.\
        experiment_results['Experiment00001']['results'].values()
    assert any(x.title == 'Number of targets' for x in metrics_results)
    assert any(x.title == 'Number of tracks' for x in metrics_results)
    assert any(x.title == 'Track-to-target ratio' for x in metrics_results)
    assert any(x.title == 'SIAP A' for x in metrics_results)
    assert any(x.title == 'SIAP C' for x in metrics_results)
    assert any(x.title == 'SIAP S' for x in metrics_results)
    assert any(x.title == 'SIAP LS' for x in metrics_results)
    assert any(x.title == 'SIAP LT' for x in metrics_results)
    assert any(x.title == 'OSPA distances' for x in metrics_results)


def test_run_manager_common_simulator(base_tracker):
    # tests a Run Manager that is given two Trackers which use a shared data
    # simulator for 'tacker.detector' - should use the same simulated data

    # -------------------------------------------------------
    #   create two standard Trackers and modify them
    #   to have 'detector' defined elsewhere
    # -------------------------------------------------------
    local_tracker = base_tracker.get_base_tracker()
    local_tracker_copy = copy.deepcopy(local_tracker['tracker01'])
    local_tracker['tracker02'] = local_tracker_copy

    detection_sim = local_tracker['tracker01'].detector

    local_tracker['tracker01'].detector = None
    local_tracker['tracker02'].detector = None

    local_tracker['components'] = \
        {"detector": detection_sim,
         "tracker01.detector": "detector",
         "tracker02.detector": "detector"}

    # -----------------------------------------------------------
    #   run the Tracker - two experiments with identical data
    # -----------------------------------------------------------
    local_run_manager = SimpleRunManager(run_checks=True,
                                         illegal_attribute_values=[None])
    local_run_manager.load_config(object=local_tracker)

    local_run_manager.run_experiment(save_detections_tracks=True)

    # -------------------------------------------------------------------
    # check that all of the experiment results were generated correctly
    # -------------------------------------------------------------------
    for experiment_name in ['Experiment00001', 'Experiment00002']:
        metrics_results = local_run_manager.\
            experiment_results[experiment_name]['results'].values()
        assert any(x.title == 'Number of targets' for x in metrics_results)
        assert any(x.title == 'Number of tracks' for x in metrics_results)
        assert any(x.title == 'Track-to-target ratio' for x in metrics_results)
        assert any(x.title == 'SIAP A' for x in metrics_results)
        assert any(x.title == 'SIAP C' for x in metrics_results)
        assert any(x.title == 'SIAP S' for x in metrics_results)
        assert any(x.title == 'SIAP LS' for x in metrics_results)
        assert any(x.title == 'SIAP LT' for x in metrics_results)
        assert any(x.title == 'OSPA distances' for x in metrics_results)

    # -------------------------------------------------------------------
    # check that the two Trackers operated on the same data
    # -------------------------------------------------------------------
    tracker_01_detections = local_run_manager.\
        experiment_results['Experiment00001']['detections']
    tracker_02_detections = local_run_manager.\
        experiment_results['Experiment00002']['detections']
    tracker_01_groundtruth = local_run_manager.\
        experiment_results['Experiment00001']['groundtruth']
    tracker_02_groundtruth = local_run_manager.\
        experiment_results['Experiment00002']['groundtruth']

    assert len(tracker_01_detections) == len(tracker_02_detections)
    assert len(tracker_01_groundtruth) == len(tracker_02_groundtruth)


def test_run_manager_error_cases(base_tracker):
    # tests several error cases

    # -------------------------------------------------
    #   test case where 'config' is not a dict()
    # -------------------------------------------------
    with pytest.raises(ValueError):
        local_run_manager = SimpleRunManager(run_checks=True,
                                             illegal_attribute_values=3)
        local_run_manager.load_config(object=list())

    # --------------------------------------------------------------
    #   test case where 'config' read from a file is not a dict()
    # --------------------------------------------------------------
    file = tempfile.NamedTemporaryFile(prefix="runmanager_test_", delete=False)
    file.close()
    f = open(file.name, 'w')
    f.write('abcdef')
    f.close()

    with pytest.raises(ValueError):
        local_run_manager = SimpleRunManager(run_checks=True,
                                             illegal_attribute_values=[None])
        local_run_manager.load_config(filename=file.name)

    file.close()
    os.remove(file.name)

    # ----------------------------------------------------------
    #   test case where 'config' not provided to Run Manager
    # ----------------------------------------------------------
    with pytest.raises(ValueError):
        local_run_manager = SimpleRunManager(run_checks=True,
                                             illegal_attribute_values=3)
        local_run_manager.load_config()

    # ----------------------------------------------------------------------
    #   test case where 'associator' not provided in 'metrics_conditions'
    # ----------------------------------------------------------------------
    local_tracker = base_tracker.get_base_tracker()

    del local_tracker['metrics']['associator']

    with pytest.raises(SystemExit):
        local_run_manager = SimpleRunManager(run_checks=True,
                                             illegal_attribute_values=[None])
        local_run_manager.load_config(object=local_tracker)
        local_run_manager.run_experiment()


def test_run_manager_monte_carlo_multiple_variant_error(base_tracker):
    # tests whether the Run Manager correctly rejects Monte Carlo
    # experiments that have more than one variant (only a single
    # variant allowed for a Monte Carlo experiment)

    # ------------------------------------------------------------------------
    #   create a standard Tracker and modify it to be a Monte Carlo experiment
    #   and have several variants
    # ------------------------------------------------------------------------
    local_tracker = base_tracker.get_base_tracker()

    deleter01 = CovarianceBasedDeleter(covar_trace_thresh=1E3)
    deleter02 = UpdateTimeStepsDeleter(5)

    local_tracker['components'] = \
        {"tracker01.deleter": iter([deleter01, deleter02])}

    local_tracker['conditions'] = {"monte_carlo": 3}

    # --------------------------------------------------------------------
    #   Test should fail because this is a Monte Carlo experiment with
    #   multiple variants (only one variant allowed for Monte Carlo)
    # --------------------------------------------------------------------
    local_run_manager = SimpleRunManager(run_checks=True,
                                         illegal_attribute_values=[None])
    local_run_manager.load_config(object=local_tracker)

    with pytest.raises(SystemExit):
        local_run_manager.run_experiment()


def test_run_manager_monte_carlo_not_simulated_data_error(base_tracker):
    # tests whether Run Manager correctly prohibits Monte Carlo experiments
    # that use other than simulated data

    # ------------------------------------------------------------------------
    #   create a standard Tracker and modify it to be a Monte Carlo experiment
    # ------------------------------------------------------------------------
    local_tracker = base_tracker.get_base_tracker()

    local_tracker['conditions'] = {"monte_carlo": 3}

    # -------------------------------------------------------------------
    #   create a "data file" for detections/groundtruth to be read from
    # -------------------------------------------------------------------
    file = tempfile.NamedTemporaryFile(prefix="runmanager_test_", delete=False)
    file.close()
    f = open(file.name, 'w')
    f.write('abcdef')
    f.close()

    # ------------------------------------------------------------------------
    #   set local_tracker['tracker01'].detector.groundtruth to a file reader
    #   rather than a data simulator
    # ------------------------------------------------------------------------
    local_tracker['tracker01'].detector.groundtruth = \
        JSON_AISDetectionReader(path=file.name)

    with pytest.raises(SystemExit):
        local_run_manager = SimpleRunManager(run_checks=True,
                                             illegal_attribute_values=[None])
        local_run_manager.load_config(object=local_tracker)
        local_run_manager.run_experiment()

    del local_tracker['tracker01'].detector.groundtruth

    # ------------------------------------------------------------------------
    #   set local_tracker['tracker01'].detector to a file reader rather than
    #   a data simulator
    # ------------------------------------------------------------------------
    local_tracker['tracker01'].detector = \
        JSON_AISDetectionReader(path=file.name)

    with pytest.raises(SystemExit):
        local_run_manager = SimpleRunManager(run_checks=True,
                                             illegal_attribute_values=[None])
        local_run_manager.load_config(object=local_tracker)
        local_run_manager.run_experiment()

    del local_tracker['tracker01'].detector

    # Cleanup
    file.close()
    os.remove(file.name)


def cleanup():
    # cleanup - remove the 'run_manager_results' folder
    shutil.rmtree(output_folder)
