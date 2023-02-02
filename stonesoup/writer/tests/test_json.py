from textwrap import dedent

import pytest

from ..json import JSONParameterWriter


def test_json_parameter(tmpdir):
    filename = tmpdir.join("parameters.json")
    with JSONParameterWriter(filename.strpath) as writer:
        writer.add_configuration()
        writer.add_parameter([2, 2, 2, 2],
                             "tracker.initiator.initiator.prior_state.state_vector",
                             "StateVector",
                             [0, 0, 0, 0],
                             [1000, 100, 100, 100])
        writer.write()

    with filename.open('r') as json_file:
        generated_json = json_file.read()
        print(generated_json)

    expected_json = dedent("""\
{
    "configuration": {
        "proc_num": 1,
        "runs_num": 10
    },
    "parameters": [
        {
            "path": "tracker.initiator.initiator.prior_state.state_vector",
            "type": "StateVector",
            "value_min": [
                0,
                0,
                0,
                0
            ],
            "value_max": [
                1000,
                100,
                100,
                100
            ],
            "n_samples": [
                2,
                2,
                2,
                2
            ]
        }
    ]
}""")
    assert generated_json == expected_json


def test_json_parameter_sv(tmpdir):
    filename = tmpdir.join("parameters.json")
    with JSONParameterWriter(filename.strpath) as writer:
        writer.add_configuration()
        writer.add_parameter(2,
                             "tracker.initiator.initiator.prior_state.state_vector",
                             "StateVector",
                             [0, 0, 0, 0],
                             [1000, 100, 100, 100])
        writer.write()

    with filename.open('r') as json_file:
        generated_json = json_file.read()

    expected_json = dedent("""\
{
    "configuration": {
        "proc_num": 1,
        "runs_num": 10
    },
    "parameters": [
        {
            "path": "tracker.initiator.initiator.prior_state.state_vector",
            "type": "StateVector",
            "value_min": [
                0,
                0,
                0,
                0
            ],
            "value_max": [
                1000,
                100,
                100,
                100
            ],
            "n_samples": [
                0,
                0,
                0,
                0
            ]
        }
    ]
}""")
    assert generated_json == expected_json


def test_json_parameter_with_2parameters(tmpdir):
    filename = tmpdir.join("parameters.json")
    with JSONParameterWriter(filename.strpath) as writer:
        writer.add_configuration()
        writer.add_parameter([2, 2, 2, 2],
                             "tracker.initiator.initiator.prior_state.state_vector",
                             "StateVector",
                             [0, 0, 0, 0],
                             [1000, 100, 100, 100])
        writer.add_parameter(7,
                             "tracker.initiator.deleter.covar_trace_thresh",
                             "float",
                             1000,
                             10000)
        writer.write()

    with filename.open('r') as json_file:
        generated_json = json_file.read()
        print(generated_json)

    expected_json = dedent("""\
{
    "configuration": {
        "proc_num": 1,
        "runs_num": 10
    },
    "parameters": [
        {
            "path": "tracker.initiator.initiator.prior_state.state_vector",
            "type": "StateVector",
            "value_min": [
                0,
                0,
                0,
                0
            ],
            "value_max": [
                1000,
                100,
                100,
                100
            ],
            "n_samples": [
                2,
                2,
                2,
                2
            ]
        },
        {
            "path": "tracker.initiator.deleter.covar_trace_thresh",
            "type": "float",
            "value_min": 1000,
            "value_max": 10000,
            "n_samples": 5
        }
    ]
}""")
    assert generated_json == expected_json


def test_bad_parameter(tmpdir):
    filename = tmpdir.join("parameters.json")
    with pytest.raises(ValueError, match="Minimum and Maximum arrays must be the same dimension"):
        with JSONParameterWriter(filename.strpath) as writer:
            writer.add_parameter([2, 2, 2, 2],
                                 "tracker.initiator.initiator.prior_state.state_vector",
                                 "StateVector",
                                 [0, 0, 0, 0],
                                 [1000, 100, 100, 100, 0])
