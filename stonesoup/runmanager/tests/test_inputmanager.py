import pytest

from stonesoup.types import array
import numpy as np
from datetime import datetime, timedelta

from ..inputmanager import InputManager

IManager = InputManager()

def test_set_stateVector():

    test_list_state_vector1 = [[1, 2, 3]]
    test_list_state_vector2 = [array.StateVector([4, 5, 6])]

    test_vector_list1 = IManager.set_stateVector(test_list_state_vector1)
    test_vector_list2 = IManager.set_stateVector(test_list_state_vector2)

    assert type(test_vector_list1) is list
    assert type(test_vector_list2) is list

    assert type(test_vector_list1[0]) is array.StateVector
    assert type(test_vector_list2[0]) is array.StateVector

def test_set_int():

    test_input_int1 = IManager.set_int(1)
    test_input_int2 = IManager.set_int(2.0)
    test_input_int3 = IManager.set_int("3")

    assert type(test_input_int1) is int
    assert type(test_input_int2) is int
    assert type(test_input_int3) is int

def test_set_float():

    test_input_float1 = IManager.set_float(1)
    test_input_float2 = IManager.set_float(2.0)
    test_input_float3 = IManager.set_float("3")

    assert type(test_input_float1) is float
    assert type(test_input_float2) is float
    assert type(test_input_float3) is float

def test_set_covariance():

    test_covar1 = IManager.set_covariance([[1, 2, 3]])
    test_covar2 = IManager.set_covariance(array.CovarianceMatrix([[4, 5, 6]]))

    assert type(test_covar1) is np.ndarray
    assert type(test_covar2) is np.ndarray

    assert type(test_covar1[0]) is np.ndarray
    assert type(test_covar2[0]) is np.ndarray

def test_set_tuple():
    test_list_tuple1 = [(1,2), (3,4)]
    test_list_tuple2 = [(1.0,"2"), ([3],True)]
    test_list_tuple3 = ((8,9),(10,11),(12,13,14))

    test_tuple_list1 = IManager.set_tuple(test_list_tuple1)
    test_tuple_list2 = IManager.set_tuple(test_list_tuple2)
    test_tuple_list3 = IManager.set_tuple(test_list_tuple3)

    assert type(test_tuple_list1) is list
    assert type(test_tuple_list2) is list
    assert type(test_tuple_list3) is list

    assert type(test_tuple_list1[0]) is tuple
    assert type(test_tuple_list2[0]) is tuple
    assert type(test_tuple_list3[0]) is tuple

def test_set_bool():
    # TODO
    pass

def test_set_ndArray():
    # TODO
    pass

def test_set_time_delta():

    test_datetime = 1
    test_time_delta = IManager.set_time_delta(test_datetime)
    assert type(test_time_delta) is timedelta

def test_set_coordinate_system():
    # TODO
    pass

def test_set_probability():
    # TODO
    pass

def test_generate_parameters_combinations():

    str_datetime = str(datetime.now())

    # Issue with StateVector, CovarianceMatrix, Tuple, ndarray
    test_params = [{'path': 'path_name' , 'var_name': 'var_name' , 'type': 'StateVector',
                     'value_min': [1, 1, 1], 'value_max': [3, 4, 5], 'n_samples': 3},
                    {'path': 'path.name2' , 'var_name': 'var_name2' , 'type': 'int',
                     'value_min': 500, 'value_max': 700, 'n_samples': 4},
                    {'path': 'path.name.3' , 'var_name': 'var_name3' , 'type': 'float',
                     'value_min': 20.0, 'value_max': 50.0, 'n_samples': 5}]

    test_params2 = [{'path': 'path_name4' , 'var_name': 'var_name4' , 'type': 'Probability',
                     'value_min': 1, 'value_max': 3, 'n_samples': 2},
                    {'path': 'path.name5' , 'var_name': 'var_name5' , 'type': 'bool',
                     'value_min': True, 'value_max': False, 'n_samples': 4},
                    {'path': 'pathname6' , 'var_name': 'var_name6' , 'type': 'CovarianceMatrix',
                     'value_min': [[1, 2, 3], [4, 5, 6]], 'value_max': [[7, 8, 9], [10, 11, 12]], 'n_samples': 3}]

    test_params3 = [{'path': 'pathname7' , 'var_name': 'var_name7' , 'type': 'DateTime',
                     'value_min': str_datetime, 'value_max': str_datetime, 'n_samples': 2},
                    {'path': 'pathname8' , 'var_name': 'var_name8' , 'type': 'Tuple',
                     'value_min': (1, 2), 'value_max': (4, 6), 'n_samples': 3}]

    test_params4 = [{'path': 'pathname9' , 'var_name': 'var_name9' , 'type': 'timedelta',
                     'value_min': 1, 'value_max': 3, 'n_samples': 2},
                    {'path': 'pathname10' , 'var_name': 'var_name10' , 'type': 'ndarray',
                     'value_min': np.array([1,2]), 'value_max': np.array([5,6]), 'n_samples': 3}]

    # test_params = [{'path': 'SingleTargetTracker.initiator.initiator.prior_state.velocity', 'var_name': 'velocity', 'type': 'vector', 'value_min': [1, 1, 1], 'value_max': [5, 1, 2], 'n_samples': 3}, {'path': 'SingleTargetTracker.initiator.initiator.prior_state.num_particles', 'var_name': 'num_particles', 'type': 'int', 'value_min': 500, 'value_max': 700, 'n_samples': 4}, {'path': 'SingleTargetTracker.initiator.initiator.prior_state.total_weight', 'var_name': 'total_weight', 'type': 'float', 'value_min': 20.5, 'value_max': 37.46, 'n_samples': 4}, {'path': 'SingleTargetTracker.initiator.initiator.prior_state.normalise', 'var_name': 'normalise', 'type': 'bool', 'bool': 0, 'n_samples': 4}]

    test_combo_dict1 = IManager.generate_parameters_combinations(test_params)
    test_combo_dict2 = IManager.generate_parameters_combinations(test_params2)
    test_combo_dict3 = IManager.generate_parameters_combinations(test_params3)
    test_combo_dict4 = IManager.generate_parameters_combinations(test_params4)

    # Testing for correct number of samples
    assert len(test_combo_dict1['path_name']) is (test_params[0]['n_samples'] + 2)**len(test_params[0]['value_min'])
    assert len(test_combo_dict1['path.name2']) is test_params[1]['n_samples'] + 2
    assert len(test_combo_dict1['path.name.3']) is test_params[2]['n_samples'] + 2

    assert len(test_combo_dict2['path_name4']) is test_params2[0]['n_samples'] + 2
    assert len(test_combo_dict2['path.name5']) is 2  # Boolean can only have 2, no steps between 0-1
    assert len(test_combo_dict2['pathname6']) is (test_params2[2]['n_samples'] + 2)**len(test_params2[2]['value_min'])

    assert len(test_combo_dict3['pathname7']) is 1  # Not sure about datetime combo steps?
    assert len(test_combo_dict3['pathname8']) is (test_params3[1]['n_samples'] + 2)**len(test_params3[1]['value_min'])

    assert len(test_combo_dict4['pathname9']) is test_params4[0]['n_samples'] + 2
    assert len(test_combo_dict4['pathname10']) is (test_params4[1]['n_samples'] + 2)**len(test_params4[1]['value_min'])

def test_darray_navigator():
    # Not yet used
    pass

def test_iterations():

    test_min_value = 0
    test_max_value = 100
    test_num_samples = 3
    test_iters = IManager.iterations(test_min_value, test_max_value, test_num_samples)
    assert len(test_iters) is test_num_samples + 2
    assert test_iters == [0.0, 25.0, 50.0, 75.0, 100.0]

    test_min_value1 = 100
    test_max_value1 = 50
    test_num_samples1 = 4
    test_iters1 = IManager.iterations(test_min_value1, test_max_value1, test_num_samples1)
    assert len(test_iters1) is test_num_samples1 + 2
    assert test_iters1 == [100.0, 90.0, 80.0, 70.0, 60.0, 50.0]

    test_min_value2 = 50
    test_max_value2 = 50
    test_num_samples2 = 5
    test_iters2 = IManager.iterations(test_min_value2, test_max_value2, test_num_samples2)
    assert len(test_iters2) is 1
    assert test_iters2 == [50]

def test_get_array_list():

    test_iter_container_list = [array.StateVector([1, 2, 3, 4, 5]),
                                array.StateVector([6, 7, 8, 9, 10])]
    test_n = 2
    test_array_list = IManager.get_array_list(test_iter_container_list, test_n)
    assert len(test_array_list) is len(test_iter_container_list[0]) * len(test_iter_container_list[1])
    for i in test_iter_container_list[0]:
        for j in test_iter_container_list[1]:
            assert (i, j) in test_array_list

    test_iter_container_list1 = [array.StateVector([10, 20, 30]),
                                 array.StateVector([6, 7, 8, 9, 10, 11, 12]),
                                 array.StateVector([1, 3])]
    test_n1 = 3
    test_array_list1 = IManager.get_array_list(test_iter_container_list1, test_n1)
    assert len(test_array_list1) is (len(test_iter_container_list1[0]) *
                                     len(test_iter_container_list1[1]) *
                                     len(test_iter_container_list1[2]))
    for i in test_iter_container_list1[0]:
        for j in test_iter_container_list1[1]:
            for k in test_iter_container_list1[2]:
                assert (i, j, k) in test_array_list1

    test_iter_container_list2 = [array.StateVector([1, 2, 3])]
    test_n2 = 1
    test_array_list2 = IManager.get_array_list(test_iter_container_list2, test_n2)
    assert len(test_array_list2) is len(test_iter_container_list2[0])
    for i in test_iter_container_list2[0]:
        assert i in test_array_list2

def test_get_ndarray_trackers_list():
    test_iter_list = [np.ndarray((2,2)),
                      np.ndarray((2,2))]
    test_n = 2
    test_ndarray_list = IManager.get_ndarray_trackers_list(test_iter_list, test_n)

    print(test_ndarray_list)

    assert len(test_ndarray_list) is len(test_iter_list[0]) * len(test_iter_list[1])
    assert len(test_ndarray_list[0]) is len(test_iter_list)

def test_get_covar_trackers_list():
    test_iter_list = [[1,2],[3,4],
                      [5,6],[7,8]]
    test_n = 4
    test_covar_list = IManager.get_covar_trackers_list(test_iter_list, test_n)

    print(test_covar_list)

    assert len(test_covar_list) is len(test_iter_list) * test_n
    assert len(test_covar_list[0]) is test_n

def test_generate_all_combos():
    test_trackers_dict = {'param1': [100, 125, 150, 175, 200], 'param2': [10, 20, 30]}

    test_combinations = IManager.generate_all_combos(test_trackers_dict)

    assert len(test_combinations) is len(test_trackers_dict['param1']) * len(test_trackers_dict['param2'])
    for i in test_trackers_dict['param1']:
        for j in test_trackers_dict['param2']:
            assert {'param1': i, 'param2': j} in test_combinations
