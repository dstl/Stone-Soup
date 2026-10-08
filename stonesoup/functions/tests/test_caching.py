import gc
import weakref

import numpy as np
import datetime

from ..caching import instance_lru_cache
from ...updater.kalman import KalmanUpdater
from ...models.measurement.linear import LinearGaussian
from ...types.prediction import GaussianStatePrediction
from ...predictor.kalman import KalmanPredictor
from ...models.transition.linear import ConstantVelocity
from ...types.state import GaussianState
from ...types.track import Track


def test_instance_lru_cache_hits_and_is_per_instance():
    class Toy:
        def __init__(self):
            self.calls = 0

        @instance_lru_cache()
        def compute(self, value):
            self.calls += 1
            return value * 2

    a = Toy()
    b = Toy()
    assert a.compute(3) == 6
    assert a.compute(3) == 6
    assert a.calls == 1  # second call served from cache
    assert b.compute(3) == 6
    assert b.calls == 1  # separate cache per instance
    assert Toy.compute.cache_info(a).hits == 1
    assert Toy.compute.cache_info(a).misses == 1


def test_instance_lru_cache_allows_garbage_collection():
    """Class-level functools.lru_cache on methods retains instances; ours must not."""
    class Toy:
        @instance_lru_cache()
        def compute(self, value):
            return value

    obj = Toy()
    assert obj.compute(1) == 1
    ref = weakref.ref(obj)
    del obj
    gc.collect()
    assert ref() is None


def test_kalman_updater_predict_measurement_cached_and_collectable():
    updater = KalmanUpdater(
        LinearGaussian(ndim_state=2, mapping=[0], noise_covar=np.array([[1.]])))
    prediction = GaussianStatePrediction(
        [[0.], [1.]], np.diag([1., 1.]), timestamp=datetime.datetime.now())

    meas1 = updater.predict_measurement(prediction)
    meas2 = updater.predict_measurement(prediction)
    assert meas1 is meas2

    ref = weakref.ref(updater)
    del updater, meas1, meas2, prediction
    gc.collect()
    assert ref() is None


def test_kalman_predictor_cache_still_works_and_is_collectable():
    predictor = KalmanPredictor(ConstantVelocity(noise_diff_coeff=0))
    timestamp = datetime.datetime.now()
    track = Track([GaussianState([[0.], [1.]], np.diag([1., 1.]), timestamp)])
    prediction_time = timestamp + datetime.timedelta(seconds=1)

    prediction1 = predictor.predict(track, prediction_time)
    prediction2 = predictor.predict(track, prediction_time)
    assert prediction2 is prediction1

    ref = weakref.ref(predictor)
    del predictor, prediction1, prediction2, track
    gc.collect()
    assert ref() is None
