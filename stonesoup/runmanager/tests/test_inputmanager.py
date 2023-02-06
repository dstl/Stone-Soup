import pytest

from stonesoup.types.numeric import Probability

from stonesoup.types import array
import numpy as np
from datetime import datetime, timedelta

from ..inputmanager import InputManager

IManager = InputManager()


def test_set_stateVector():

    test_list_state_vector1 = [[1, 2, 3]]
    test_list_state_vector2 = [array.StateVector([4, 5, 6])]
    test_empty_list_state_vector = []

    test_vector_list1 = IManager.set_stateVector(test_list_state_vector1)
    test_vector_list2 = IManager.set_stateVector(test_list_state_vector2)
    test_empty_vector_list = IManager.set_stateVector(test_empty_list_state_vector)

    assert type(test_vector_list1) is list
    assert type(test_vector_list2) is list
    assert type(test_empty_vector_list) is list

    assert type(test_vector_list1[0]) is array.StateVector
    assert type(test_vector_list2[0]) is array.StateVector
    assert len(test_empty_vector_list) == 0


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

    test_covar1 = IManager.set_covariance([(1, 2, 3)])
    test_covar2 = IManager.set_covariance([(4, 5, 6)])
    test_empty_covar = IManager.set_covariance([()])

    assert type(test_covar1) is list
    assert type(test_covar2) is list
    assert type(test_empty_covar) is list

    assert type(test_covar1[0]) is array.CovarianceMatrix
    assert type(test_covar2[0]) is array.CovarianceMatrix
    assert len(test_empty_covar) == 1


def test_set_tuple():
    test_list_tuple1 = [(1, 2), (3, 4)]
    test_list_tuple2 = [(1.0, "2"), ([3], True)]
    test_list_tuple3 = ((8, 9), (10, 11), (12, 13, 14))

    test_tuple_list1 = IManager.set_tuple(test_list_tuple1)
    test_tuple_list2 = IManager.set_tuple(test_list_tuple2)
    test_tuple_list3 = IManager.set_tuple(test_list_tuple3)
    test_empty_list_tuple = IManager.set_tuple([])

    assert type(test_tuple_list1) is list
    assert type(test_tuple_list2) is list
    assert type(test_tuple_list3) is list
    assert type(test_empty_list_tuple) is list

    assert type(test_tuple_list1[0]) is tuple
    assert type(test_tuple_list2[0]) is tuple
    assert type(test_tuple_list3[0]) is tuple
    assert len(test_empty_list_tuple) == 0


def test_set_bool():
    test_input_bool1 = IManager.set_bool(True)
    test_input_bool2 = IManager.set_bool(1)
    test_input_bool3 = IManager.set_bool(False)
    test_input_bool4 = IManager.set_bool(0)

    assert type(test_input_bool1) is bool
    assert type(test_input_bool2) is bool
    assert type(test_input_bool3) is bool
    assert type(test_input_bool4) is bool
    with pytest.raises(ValueError, match="input_bool must be of boolean type."):
        IManager.set_bool(0.5)


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
    test_input_prob1 = IManager.set_probability(0.1)
    test_input_prob2 = IManager.set_probability(Probability(0.2))

    assert type(test_input_prob1) is Probability
    assert type(test_input_prob2) is Probability


def test_generate_parameters_combinations():

    str_datetime = str(datetime.now())

    # Issue with StateVector, CovarianceMatrix, Tuple, ndarray
    test_params = [{'path': 'path_name', 'var_name': 'var_name', 'type': 'StateVector',
                    'value_min': [0, 0, 0], 'value_max': [100, 100, 100], 'n_samples': [1, 1, 0]},
                   {'path': 'path.name2', 'var_name': 'var_name2', 'type': 'int',
                    'value_min': 500, 'value_max': 700, 'n_samples': 4},
                   {'path': 'path.name.3', 'var_name': 'var_name3', 'type': 'float',
                    'value_min': 20.0, 'value_max': 50.0, 'n_samples': 5}]

    test_params2 = [{'path': 'path_name4', 'var_name': 'var_name4', 'type': 'Probability',
                     'value_min': 1, 'value_max': 3, 'n_samples': 2},
                    {'path': 'path.name5', 'var_name': 'var_name5', 'type': 'bool',
                     'value_min': True, 'value_max': False, 'n_samples': 4},
                    {'path': 'pathname6', 'var_name': 'var_name6', 'type': 'CovarianceMatrix',
                     'value_min': [1, 2, 3, 0], 'value_max': [10, 11, 12, 0],
                     'n_samples': [1, 0, 0, 0]}]

    test_params3 = [{'path': 'pathname7', 'var_name': 'var_name7', 'type': 'DateTime',
                     'value_min': str_datetime, 'value_max': str_datetime, 'n_samples': 2},
                    {'path': 'pathname8', 'var_name': 'var_name8', 'type': 'Tuple',
                     'value_min': (1, 2), 'value_max': (4, 6), 'n_samples': (3, 3)}]

    test_params4 = [{'path': 'pathname9', 'var_name': 'var_name9', 'type': 'timedelta',
                     'value_min': 1, 'value_max': 3, 'n_samples': 2},
                    {'path': 'pathname10', 'var_name': 'var_name10', 'type': 'ndarray',
                     'value_min': np.array([1, 2]), 'value_max': np.array([5, 6]),
                     'n_samples': [3, 3]}]

    test_combo_dict1 = IManager.generate_parameters_combinations(test_params)
    test_combo_dict2 = IManager.generate_parameters_combinations(test_params2)
    test_combo_dict3 = IManager.generate_parameters_combinations(test_params3)
    test_combo_dict4 = IManager.generate_parameters_combinations(test_params4)

    # Testing for correct number of samples
    prod_1 = test_params[0]['n_samples'][0] + 2
    prod_2 = test_params[0]['n_samples'][1] + 2
    prod_3 = test_params[0]['n_samples'][2] + 2

    assert len(test_combo_dict1['path_name']) is prod_1 * prod_2 * prod_3
    assert len(test_combo_dict1['path.name2']) is test_params[1]['n_samples'] + 2
    assert len(test_combo_dict1['path.name.3']) is test_params[2]['n_samples'] + 2

    assert len(test_combo_dict2['path_name4']) is test_params2[0]['n_samples'] + 2
    assert len(test_combo_dict2['path.name5']) == 2  # Boolean only has 2, no steps between 0-1
    assert (len(test_combo_dict2['pathname6'][0][0])**len(test_params2[2]['value_min']))

    assert len(test_combo_dict3['pathname7']) == 1
    assert (len(test_combo_dict3['pathname8']) is
            (test_params3[1]['n_samples'][0] + 2)**len(test_params3[1]['value_min']))

    assert len(test_combo_dict4['pathname9']) is test_params4[0]['n_samples'] + 2
    assert (len(test_combo_dict4['pathname10']) is
            (test_params4[1]['n_samples'][0] + 2)**len(test_params4[1]['value_min']))


def test_generate_parameters_statevector():
    test_empty_sv = {'path': 'path_name', 'var_name': 'var_name', 'type': 'StateVector',
                     'value_min': [], 'value_max': [], 'n_samples': [3]}
    test_sv_0nsamples = {'path': 'path_name', 'var_name': 'var_name', 'type': 'StateVector',
                         'value_min': [1], 'value_max': [2], 'n_samples': [0]}
    test_empty_sv_dict = IManager.generate_parameters_combinations([test_empty_sv])
    test_sv_0nsamples_dict = IManager.generate_parameters_combinations([test_sv_0nsamples])

    assert test_empty_sv_dict == {}

    assert len(test_sv_0nsamples_dict['path_name']) == 2
    assert len(test_sv_0nsamples_dict['path_name'][0]) is len(test_sv_0nsamples['value_min'])
    assert type(test_sv_0nsamples_dict['path_name'][0]) is array.StateVector


def test_generate_parameters_int():
    test_int_0nsamples = {'path': 'path_name', 'var_name': 'var_name', 'type': 'int',
                          'value_min': 1, 'value_max': 2, 'n_samples': 0}

    test_int_0nsamples_dict = IManager.generate_parameters_combinations([test_int_0nsamples])

    assert len(test_int_0nsamples_dict['path_name']) == 2
    assert test_int_0nsamples_dict['path_name'][0] == 1
    assert type(test_int_0nsamples_dict['path_name'][0]) is int


def test_generate_parameters_float():
    test_float_0nsamples = {'path': 'path_name', 'var_name': 'var_name', 'type': 'float',
                            'value_min': 1.0, 'value_max': 2.0, 'n_samples': 0}

    test_float_0nsamples_dict = IManager.generate_parameters_combinations([test_float_0nsamples])

    assert len(test_float_0nsamples_dict['path_name']) == 2
    assert test_float_0nsamples_dict['path_name'][0] == 1.0
    assert type(test_float_0nsamples_dict['path_name'][0]) is float


def test_generate_parameters_probability():
    test_prob_0nsamples = {'path': 'path_name', 'var_name': 'var_name',
                           'type': 'Probability', 'value_min': 1, 'value_max': 2,
                           'n_samples': 0}

    test_prob_0nsamples_dict = IManager.generate_parameters_combinations([test_prob_0nsamples])

    assert len(test_prob_0nsamples_dict['path_name']) == 2
    assert test_prob_0nsamples_dict['path_name'][0] == Probability(1.0)
    assert type(test_prob_0nsamples_dict['path_name'][0]) is Probability


def test_generate_parameters_bool():
    test_bool_0nsamples = {'path': 'path.name5', 'var_name': 'var_name5', 'type': 'bool',
                           'value_min': True, 'value_max': False, 'n_samples': 0}

    test_bool_0nsamples_dict = IManager.generate_parameters_combinations([test_bool_0nsamples])

    assert len(test_bool_0nsamples_dict['path.name5']) == 2
    assert test_bool_0nsamples_dict['path.name5'][0] is True
    assert type(test_bool_0nsamples_dict['path.name5'][0]) is bool


def test_generate_parameters_covariancematrix():
    test_empty_cv = {'path': 'pathname6', 'var_name': 'var_name6', 'type': 'CovarianceMatrix',
                     'value_min': [],
                     'value_max': [], 'n_samples': []}
    test_cv_0nsamples = {'path': 'pathname6', 'var_name': 'var_name6', 'type': 'CovarianceMatrix',
                         'value_min': [1, 1, 1, 1], 'value_max': [2, 2, 2, 2],
                         'n_samples': [1, 0, 0, 0]}

    test_empty_cv_dict = IManager.generate_parameters_combinations([test_empty_cv])
    test_cv_0nsamples_dict = IManager.generate_parameters_combinations([test_cv_0nsamples])
    print(len(test_cv_0nsamples_dict))
    assert test_empty_cv_dict == {}

    assert len(test_cv_0nsamples_dict['pathname6']) == 24
    assert len(test_cv_0nsamples_dict['pathname6'][0]) is len(test_cv_0nsamples['value_min'])
    assert type(test_cv_0nsamples_dict['pathname6'][0]) is array.CovarianceMatrix


def test_generate_parameters_datetime():
    str_datetime = str(datetime.now())

    test_dt_0nsamples = {'path': 'pathname7', 'var_name': 'var_name7', 'type': 'DateTime',
                         'value_min': str_datetime, 'value_max': str_datetime, 'n_samples': 0}

    test_dt_0nsamples_dict = IManager.generate_parameters_combinations([test_dt_0nsamples])
    assert len(test_dt_0nsamples_dict['pathname7']) == 1
    assert str(test_dt_0nsamples_dict['pathname7'][0]) == str_datetime
    assert type(test_dt_0nsamples_dict['pathname7'][0]) is datetime


def test_generate_parameters_tuple():
    test_empty_tuple = {'path': 'pathname8', 'var_name': 'var_name8', 'type': 'Tuple',
                        'value_min': (), 'value_max': (), 'n_samples': 2}
    test_tuple_0nsamples = {'path': 'pathname8', 'var_name': 'var_name8', 'type': 'Tuple',
                            'value_min': (1, 1), 'value_max': (2, 2), 'n_samples': (0, 0)}

    test_empty_tuple_dict = IManager.generate_parameters_combinations([test_empty_tuple])
    test_tuple_0nsamples_dict = IManager.generate_parameters_combinations([test_tuple_0nsamples])

    assert test_empty_tuple_dict == {}

    assert len(test_tuple_0nsamples_dict['pathname8']) == 4
    assert len(test_tuple_0nsamples_dict['pathname8'][0]) is len(test_tuple_0nsamples['value_min'])
    assert type(test_tuple_0nsamples_dict['pathname8'][0]) is tuple


# def test_generate_parameters_list():
# TO BE DONE
#     test_empty_list = {'path': 'pathname', 'var_name': 'var_name', 'type': 'list',
#                        'value_min': [], 'value_max': [], 'n_samples': 2}
#     test_list_0nsamples = {'path': 'pathname', 'var_name': 'var_name', 'type': 'list',
#                            'value_min': [1, 1], 'value_max': [2, 2], 'n_samples': [0, 0]}

#     test_empty_list_dict = IManager.generate_parameters_combinations([test_empty_list])
#     test_list_0nsamples_dict = IManager.generate_parameters_combinations([test_list_0nsamples])

#     assert test_empty_list_dict == {}
#     assert len(test_list_0nsamples_dict['pathname']) == 1
#     assert len(test_list_0nsamples_dict['pathname'][0]) is len(test_list_0nsamples['value_min'])
#     assert type(test_list_0nsamples_dict['pathname'][0]) is list


def test_generate_parameters_timedelta():
    test_td_0nsamples = {'path': 'pathname9', 'var_name': 'var_name9', 'type': 'timedelta',
                         'value_min': 1, 'value_max': 2, 'n_samples': 0}

    test_td_0nsamples_dict = IManager.generate_parameters_combinations([test_td_0nsamples])
    assert len(test_td_0nsamples_dict['pathname9']) == 2
    assert str(test_td_0nsamples_dict['pathname9'][0]) == '1 day, 0:00:00'
    assert type(test_td_0nsamples_dict['pathname9'][0]) is timedelta


def test_generate_parameters_ndarray():
    test_empty_ndarray = {'path': 'pathname10', 'var_name': 'var_name10', 'type': 'ndarray',
                          'value_min': np.array([]), 'value_max': np.array([]), 'n_samples': 3}
    test_ndarr_0nsamples = {'path': 'pathname10', 'var_name': 'var_name10', 'type': 'ndarray',
                            'value_min': np.array([1, 2]), 'value_max': np.array([5, 6]),
                            'n_samples': [0, 0]}

    test_empty_ndarray_dict = IManager.generate_parameters_combinations([test_empty_ndarray])
    test_ndarr_0nsamples_dict = IManager.generate_parameters_combinations([test_ndarr_0nsamples])

    assert test_empty_ndarray_dict == {}

    assert len(test_ndarr_0nsamples_dict['pathname10']) == 4
    assert (len(test_ndarr_0nsamples_dict['pathname10'][0]) is
            len(test_ndarr_0nsamples['value_min']))
    assert type(test_ndarr_0nsamples_dict['pathname10'][0]) is np.ndarray


def test_generate_parameters_None():
    test_nonetype = {'path': 'path_name', 'var_name': 'var_name', 'type': 'None',
                     'value_min': None, 'value_max': None, 'n_samples': 0}

    test_nonetype_dict = IManager.generate_parameters_combinations([test_nonetype])
    assert test_nonetype_dict == {}


def test_generate_parameters_no_params():
    # Empty parameters
    empty_params = []
    empty_combi_dict = IManager.generate_parameters_combinations(empty_params)
    assert len(empty_combi_dict) == 0

    empty_params2 = [{}]
    empty_combi_dict2 = IManager.generate_parameters_combinations(empty_params2)
    assert len(empty_combi_dict2) == 0


def test_generate_parameters_large_nsamples():
    # Large n samples value
    large_n = 999999
    test_large_nsamples = {'path': 'path_name', 'var_name': 'var_name', 'type': 'int',
                           'value_min': 1, 'value_max': 2, 'n_samples': large_n}
    test_large_nsamples_dict = IManager.generate_parameters_combinations([test_large_nsamples])

    assert len(test_large_nsamples_dict['path_name']) == large_n + 2
    assert test_large_nsamples_dict['path_name'][0] == 1


def test_generate_parameters_negative_nsamples():
    # Negative n samples
    test_negative_nsamples = {'path': 'path_name', 'var_name': 'var_name', 'type': 'int',
                              'value_min': 1, 'value_max': 2, 'n_samples': -1}

    negative_nsamples_dict = IManager.generate_parameters_combinations([test_negative_nsamples])
    assert len(negative_nsamples_dict) == 1


def test_darray_navigator():
    # Not yet used
    pass


def test_iterations():

    test_min_value = 0
    test_max_value = 100
    test_num_samples = 3
    test_iters = IManager.iterations(test_min_value, test_max_value, test_num_samples)
    assert len(test_iters) is test_num_samples + 2
    assert test_iters == [0.0, 25.0, 50.0, 75.0, 100]

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
    assert len(test_iters2) == 1
    assert test_iters2 == [50]

    test_min_value3 = None
    test_max_value3 = None
    test_num_samples3 = 0
    test_iters3 = IManager.iterations(test_min_value3, test_max_value3, test_num_samples3)
    assert len(test_iters3) == 1
    assert test_iters3 == [None]

    test_min_value4 = -20.0
    test_max_value4 = 10
    test_num_samples4 = 1
    test_iters4 = IManager.iterations(test_min_value4, test_max_value4, test_num_samples4)
    assert len(test_iters4) == 3
    assert test_iters4 == [-20.0, -5.0, 10]

    test_min_value5 = -20.0
    test_max_value5 = 10
    test_num_samples5 = -1
    test_iters5 = IManager.iterations(test_min_value5, test_max_value5, test_num_samples5)
    assert len(test_iters5) == 2
    assert test_iters5 == [-20.0, 10]


def test_get_array_list():

    test_iter_container_list = [array.StateVector([1, 2, 3, 4, 5]),
                                array.StateVector([6, 7, 8, 9, 10])]
    test_n = 2
    test_array_list = IManager.get_array_list(test_iter_container_list, test_n)
    assert len(test_array_list) is (len(test_iter_container_list[0]) *
                                    len(test_iter_container_list[1]))
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

    test_empty_iter_container_list = []
    test_empty_array_list = IManager.get_array_list(test_empty_iter_container_list, 0)
    assert len(test_empty_array_list) == 1
    assert test_empty_array_list[0] == ()


def test_get_ndarray_trackers_list():
    test_iter_list = [np.ndarray((2, 2)),
                      np.ndarray((2, 2))]
    test_n = 2
    test_ndarray_list = IManager.get_ndarray_trackers_list(test_iter_list, test_n)

    assert len(test_ndarray_list) is len(test_iter_list[0]) * len(test_iter_list[1])
    assert len(test_ndarray_list[0]) is len(test_iter_list)

    test_empty_iter_list = []
    test_empty_ndarray_list = IManager.get_ndarray_trackers_list(test_empty_iter_list, 0)

    assert len(test_empty_ndarray_list) == 1
    assert len(test_empty_ndarray_list[0]) is len(test_empty_iter_list)


def test_get_covar_trackers_list():
    test_iter_list = [[1, 2], [3, 4],
                      [5, 6], [7, 8]]
    test_n = 4
    test_covar_list = IManager.get_covar_trackers_list(test_iter_list, test_n)

    assert len(test_covar_list) is len(test_iter_list) * test_n
    assert len(test_covar_list[0]) is test_n

    test_empty_iter_list = []
    test_empty_covar_list = IManager.get_covar_trackers_list(test_empty_iter_list, 0)

    assert len(test_empty_covar_list) == 1
    assert len(test_empty_covar_list[0]) is len(test_empty_iter_list)


def test_generate_all_combos():
    test_trackers_dict = {'param1': [100, 125, 150, 175, 200], 'param2': [10, 20, 30]}

    test_combinations = IManager.generate_all_combos(test_trackers_dict)

    assert len(test_combinations) is (len(test_trackers_dict['param1']) *
                                      len(test_trackers_dict['param2']))
    for i in test_trackers_dict['param1']:
        for j in test_trackers_dict['param2']:
            assert {'param1': i, 'param2': j} in test_combinations

    test_empty_trackers_dict = {'param1': [], 'param2': []}

    test_empty_combinations = IManager.generate_all_combos(test_empty_trackers_dict)

    assert len(test_empty_combinations) is (len(test_empty_trackers_dict['param1']) *
                                            len(test_empty_trackers_dict['param2']))
    for i in test_empty_trackers_dict['param1']:
        for j in test_empty_trackers_dict['param2']:
            assert {'param1': i, 'param2': j} in test_empty_combinations

    test_empty_trackers_dict2 = {'param1': [], 'param2': [1]}

    test_empty_combinations2 = IManager.generate_all_combos(test_empty_trackers_dict2)

    assert len(test_empty_combinations2) is (len(test_empty_trackers_dict2['param1']) *
                                             len(test_empty_trackers_dict2['param2']))
    for i in test_empty_trackers_dict2['param1']:
        for j in test_empty_trackers_dict2['param2']:
            assert {'param1': i, 'param2': j} in test_empty_combinations2

    test_empty_trackers_dict3 = {'param1': []}

    test_empty_combinations3 = IManager.generate_all_combos(test_empty_trackers_dict3)

    assert len(test_empty_combinations3) is len(test_empty_trackers_dict3['param1'])
    for i in test_empty_trackers_dict3['param1']:
        assert {'param1': i} in test_empty_combinations3

    test_one_trackers_dict = {'param1': [1]}

    test_one_combinations = IManager.generate_all_combos(test_one_trackers_dict)

    assert len(test_one_combinations) is len(test_one_trackers_dict['param1'])
    for i in test_one_trackers_dict['param1']:
        assert {'param1': i} in test_one_combinations
