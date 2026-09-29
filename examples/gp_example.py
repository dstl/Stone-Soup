"""
Stone Soup Gaussian Process Transition Model Example
====================================================


"""

from datetime import datetime, timedelta

from stonesoup.types.array import StateVector
from stonesoup.types.state import State
from stonesoup.types.track import Track
from stonesoup.models.transition.gaussian_process import SimpleGaussianProcess

init_time = datetime.now()
init_state = State(StateVector([1, 0, 0]), timestamp=init_time+timedelta(seconds=1))
track = Track([init_state])

gp = SimpleGaussianProcess(num_lags=3, sigma=1, start_time=init_time)

# Time: 1.2 sec
# =====================================================
time_interval = timedelta(seconds=0.2)
new_timestamp = track.state.timestamp+time_interval
# Compute transition and covariance matrices (just for display)
F = gp.matrix(track, time_interval=time_interval)
Q = gp.covar(track, time_interval=time_interval)
# Run the track through the model to compute new state vector
sv = gp.function(track, time_interval=time_interval)
# Append state to track
track.append(State(sv, timestamp=new_timestamp))

# Time: 2 sec (1.2 + 0.8)
# =====================================================
time_interval = timedelta(seconds=0.8)
new_timestamp = track.state.timestamp+time_interval
# Compute transition and covariance matrices (just for display)
F = gp.matrix(track, time_interval=time_interval)
Q = gp.covar(track, time_interval=time_interval)
# Run the track through the model to compute new state vector
sv = gp.function(track, time_interval=time_interval)
# Append state to track
track.append(State(sv, timestamp=new_timestamp))

# Time: 3 sec (2 + 1)
# =====================================================
time_interval = timedelta(seconds=1)
new_timestamp = track.state.timestamp+time_interval
# Compute transition and covariance matrices (just for display)
F = gp.matrix(track, time_interval=time_interval)
Q = gp.covar(track, time_interval=time_interval)
# Run the track through the model to compute new state vector
sv = gp.function(track, time_interval=time_interval)
# Append state to track
track.append(State(sv, timestamp=new_timestamp))

a=2

