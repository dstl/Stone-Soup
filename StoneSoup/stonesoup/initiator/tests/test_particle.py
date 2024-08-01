from datetime import datetime

import numpy as np
import pytest

from stonesoup.predictor.particle import ParticlePredictor
from stonesoup.resampler.particle import SystematicResampler
from stonesoup.types.array import StateVector, StateVectors
from stonesoup.types.detection import TrueDetection, Clutter
from stonesoup.types.prediction import Prediction
from stonesoup.types.state import ParticleState
from stonesoup.types.update import Update
from stonesoup.updater.particle import ParticleUpdater
from stonesoup.initiator.particle import SMCPHDInitiator


@pytest.fixture()
def dummy_predictor():
    class DummyPredictor(ParticlePredictor):
        def predict(self, prior, control_input=None, timestamp=None, **kwargs):
            return Prediction.from_state(prior,
                                         parent=prior,
                                         state_vector=prior.state_vector,
                                         timestamp=timestamp,
                                         transition_model=self.transition_model,
                                         prior=prior)

        @property
        def transition_model(self):
            return None

    return DummyPredictor()


@pytest.fixture(
    params=[
        SystematicResampler(),
        None
    ],
    ids=['systematic_resampler', 'no_resampler']
)
def dummy_updater(request):
    class DummyUpdater(ParticleUpdater):
        resampler = request.param

        def predict_measurement(self, state_prediction,
                                measurement_model=None, **kwargs):
            return None

        def update(self, hypothesis, **kwargs):
            prediction = hypothesis[0].prediction
            return Update.from_state(state=prediction,
                                     hypothesis=hypothesis,
                                     timestamp=prediction.timestamp)

        def get_log_weights_per_hypothesis(self, hypothesis):
            prediction = hypothesis[0].prediction
            num_samples = prediction.state_vector.shape[1]
            log_weights_per_hyp = np.full((num_samples, len(hypothesis)), -np.inf)
            for i, hyp in enumerate(hypothesis):
                if not hyp:
                    # Set missed detection high to ensure test covers this case
                    log_weights_per_hyp[:, 0] = np.log(.9 / num_samples)
                    continue
                if isinstance(hyp.measurement, TrueDetection):
                    log_weights_per_hyp[:, i] = np.log(.8 / num_samples)
                else:
                    log_weights_per_hyp[:, i] = np.log(.1 / num_samples)
            return log_weights_per_hyp

        @property
        def measurement_model(self):
            return None

    return DummyUpdater()


@pytest.mark.parametrize(
    'num_particles, num_track_samples, threshold, resampler',
    [(100, 150, 0.7, SystematicResampler()),
     (100, 200, 0.7, None),
     (100, 200, 0.9, None)]
)
def test_smc_phd_initiator(num_particles, num_track_samples, threshold, resampler,
                           dummy_predictor, dummy_updater):
    timestamp = datetime.now()
    detections = {
        TrueDetection(StateVector([0]), timestamp=timestamp, groundtruth_path=None),
        Clutter(StateVector([5]), timestamp=timestamp),
        TrueDetection(StateVector([10]), timestamp=timestamp, groundtruth_path=None),
    }

    prior_sv = StateVectors([[i for i in range(num_particles)]])
    prior = ParticleState(state_vector=prior_sv,
                          weight=np.array([1 / num_particles] * num_particles),
                          timestamp=timestamp)

    # Test if error is raised when no resampler is specified and updater has no resampler
    if resampler is None and dummy_updater.resampler is None:
        with pytest.raises(ValueError):
            initiator = SMCPHDInitiator(prior_state=prior, predictor=dummy_predictor,
                                        updater=dummy_updater, threshold=threshold,
                                        num_track_samples=num_track_samples,
                                        resampler=resampler)
        return

    initiator = SMCPHDInitiator(prior_state=prior, predictor=dummy_predictor,
                                updater=dummy_updater, threshold=threshold,
                                num_track_samples=num_track_samples,
                                resampler=resampler)

    tracks = initiator.initiate(detections, timestamp)

    if threshold > 0.8:
        assert len(tracks) == 0
    else:
        assert len(tracks) == 2

    for track in tracks:
        assert len(track.state) == num_track_samples
        assert float(np.sum(track.state.weight)) == pytest.approx(1, abs=1e-10)
