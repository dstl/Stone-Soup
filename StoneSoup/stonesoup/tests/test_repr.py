from textwrap import dedent

from stonesoup.types.base import Base
from stonesoup.types.array import StateVector
from stonesoup.types.track import Track
from stonesoup.types.state import State
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
                                               ConstantVelocity
from datetime import datetime, timedelta


def test_repr():
    vector = StateVector([1, 2, 3, 4])
    exp_repr = dedent('''\
        StateVector([[1],
                     [2],
                     [3],
                     [4]])''')
    act_repr = repr(vector)
    assert exp_repr == act_repr
    track = Track([State(vector, timestamp=0),
                   State(StateVector([1.5, 3, 6, 9, 15]), timestamp=1)],
                  id=1)
    exp_repr2 = dedent('''\
        Track(
            states=[State(
                       state_vector=StateVector([[1],
                                                 [2],
                                                 [3],
                                                 [4]]),
                       timestamp=0),
                    State(
                       state_vector=StateVector([[ 1.5],
                                                 [ 3. ],
                                                 [ 6. ],
                                                 [ 9. ],
                                                 [15. ]]),
                       timestamp=1)],
            id=1,
            init_metadata={})''')
    act_repr2 = repr(track)
    assert exp_repr2 == act_repr2
    too_big = State([0] * 500000)  # This should not print in its entirety as it is far too large
    too_big_repr = repr(too_big)
    assert '...' in too_big_repr and len(too_big_repr) < 500  # Shortened and < 500 characters
    start_time = datetime(2021, 12, 9, 9, 32, 25, 449038)
    transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0),
                                                              ConstantVelocity(0)])
    truth = GroundTruthPath([GroundTruthState([0, 1, 0, 1], timestamp=start_time)])
    num_steps = 100
    for k in range(1, num_steps + 1):
        truth.append(GroundTruthState(
            transition_model.function(truth[k-1], noise=False, time_interval=timedelta(seconds=1)),
            timestamp=start_time+timedelta(seconds=k)))
    act_repr3 = repr(truth)
    exp_repr3_part = '''
            ...
            ...
            ...
            GroundTruthState(
               state_vector=StateVector([[96.],
                                         [ 1.],
                                         [96.],
                                         [ 1.]]),
               timestamp=2021-12-09 09:34:01.449038,
               metadata={}),'''
    # for i in range(0, len(exp_repr3_part)-1):
    #     if not exp_repr3_part[:i+1] in act_repr3:  # Useful for finding the cause of an error
    #         print(exp_repr3_part[:i+1])
    # print(act_repr3)

    # Assuming self.maxlist is set to 10 in BaseRepr
    assert exp_repr3_part in act_repr3 and len(act_repr3) < 5000

    # Check that whitespace_remove works
    large_whitespace = ' '*200
    act_repr4 = Base._repr.whitespace_remove(50, f'large {large_whitespace} whitespace')
    assert len(act_repr4) < 100 and act_repr4[-3:] == '...'
