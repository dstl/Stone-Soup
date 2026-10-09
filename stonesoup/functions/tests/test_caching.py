import copy
import datetime
import gc
import pickle
import weakref

import numpy as np

from ..caching import instance_lru_cache
from ...updater.kalman import KalmanUpdater
from ...models.measurement.linear import LinearGaussian
from ...types.prediction import GaussianStatePrediction
from ...predictor.kalman import KalmanPredictor
from ...models.transition.linear import ConstantVelocity
from ...types.state import GaussianState
from ...types.track import Track


class Toy:
    def __init__(self):
        self.calls = 0

    @instance_lru_cache()
    def compute(self, value):
        self.calls += 1
        return value * 2


def _updater_and_prediction():
    updater = KalmanUpdater(
        LinearGaussian(ndim_state=2, mapping=[0], noise_covar=np.array([[1.]])))
    prediction = GaussianStatePrediction(
        [[0.], [1.]], np.diag([1., 1.]), timestamp=datetime.datetime.now())
    return updater, prediction


def _predictor_and_track():
    predictor = KalmanPredictor(ConstantVelocity(noise_diff_coeff=0))
    timestamp = datetime.datetime.now()
    track = Track([GaussianState([[0.], [1.]], np.diag([1., 1.]), timestamp)])
    return predictor, track, timestamp + datetime.timedelta(seconds=1)


def test_instance_lru_cache_hits_and_is_per_instance():
    a = Toy()
    b = Toy()
    assert Toy.compute.cache_info(a).hits == 0
    assert a.compute(3) == 6
    assert a.compute(3) == 6
    assert a.calls == 1  # second call served from cache
    assert b.compute(3) == 6
    assert b.calls == 1  # separate cache per instance
    assert Toy.compute.cache_info(a).hits == 1
    assert Toy.compute.cache_info(a).misses == 1
    assert Toy.compute.cache_info(b).hits == 0


def test_instance_lru_cache_not_stored_on_instance():
    obj = Toy()
    obj.compute(1)
    assert set(vars(obj)) == {'calls'}
    assert obj in Toy.compute._instance_caches


def test_instance_lru_cache_clear():
    a = Toy()
    b = Toy()
    a.compute(1)
    b.compute(1)
    Toy.compute.cache_clear(a)
    assert Toy.compute.cache_info(a).currsize == 0
    assert Toy.compute.cache_info(b).currsize == 1
    Toy.compute.cache_clear()
    assert Toy.compute.cache_info(b).currsize == 0
    assert b.compute(1) == 2
    assert b.calls == 2


def test_instance_lru_cache_allows_garbage_collection():
    """Class-level functools.lru_cache on methods retains instances; ours must not."""
    obj = Toy()
    assert obj.compute(1) == 1 * 2
    ref = weakref.ref(obj)
    n_caches = len(Toy.compute._instance_caches)
    del obj
    gc.collect()
    assert ref() is None
    assert len(Toy.compute._instance_caches) == n_caches - 1


def test_kalman_updater_predict_measurement_cached_and_collectable():
    updater, prediction = _updater_and_prediction()

    meas1 = updater.predict_measurement(prediction)
    meas2 = updater.predict_measurement(prediction)
    assert meas1 is meas2
    assert KalmanUpdater.predict_measurement.cache_info(updater).hits == 1

    ref = weakref.ref(updater)
    del updater, meas1, meas2, prediction
    gc.collect()
    assert ref() is None


def test_kalman_predictor_cache_still_works_and_is_collectable():
    predictor, track, prediction_time = _predictor_and_track()

    prediction1 = predictor.predict(track, prediction_time)
    prediction2 = predictor.predict(track, prediction_time)
    assert prediction2 is prediction1
    assert KalmanPredictor.predict.cache_info(predictor).hits == 1

    ref = weakref.ref(predictor)
    del predictor, prediction1, prediction2, track
    gc.collect()
    assert ref() is None


def test_updater_pickle_and_deepcopy_after_cached_call():
    updater, prediction = _updater_and_prediction()
    expected = updater.predict_measurement(prediction)

    for new_updater in (pickle.loads(pickle.dumps(updater)), copy.deepcopy(updater)):
        assert new_updater is not updater
        meas = new_updater.predict_measurement(prediction)
        assert np.allclose(meas.state_vector, expected.state_vector)
        assert np.allclose(meas.covar, expected.covar)
        assert new_updater.predict_measurement(prediction) is meas
        info = KalmanUpdater.predict_measurement.cache_info(new_updater)
        assert info.hits == 1
        assert info.misses == 1


def test_predictor_pickle_and_deepcopy_after_cached_call():
    predictor, track, prediction_time = _predictor_and_track()
    expected = predictor.predict(track, prediction_time)

    for new_predictor in (pickle.loads(pickle.dumps(predictor)), copy.deepcopy(predictor)):
        assert new_predictor is not predictor
        prediction = new_predictor.predict(track, prediction_time)
        assert np.allclose(prediction.state_vector, expected.state_vector)
        assert np.allclose(prediction.covar, expected.covar)
        assert new_predictor.predict(track, prediction_time) is prediction
        info = KalmanPredictor.predict.cache_info(new_predictor)
        assert info.hits == 1
        assert info.misses == 1
