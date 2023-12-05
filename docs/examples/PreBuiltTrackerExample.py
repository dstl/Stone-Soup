from stonesoup.tracker.prebuilttrackers import PreBuiltSingleTargetTrackerNoClutter
from stonesoup.models.transition.linear import (
    ConstantVelocity,
    CombinedLinearGaussianTransitionModel
)
from stonesoup.sensor.radar.radar import RadarBearingRange
from stonesoup.predictor.kalman import UnscentedKalmanPredictor

detector = RadarBearingRange(ndim_state=6,
                                     position_mapping=(0, 2, 4),
                                     noise_covar=None)

ground_truth_prior = [0, 0, 5]

motion_model_noise = 0.1
target_transition_model = CombinedLinearGaussianTransitionModel(
    (ConstantVelocity(motion_model_noise), ConstantVelocity(motion_model_noise),
     ConstantVelocity(motion_model_noise)))

tracker1 = PreBuiltSingleTargetTrackerNoClutter(detector=detector,
                                                ground_truth_prior=ground_truth_prior,
                                                target_transition_model=target_transition_model)

# Do some tracking...
# Change a single component in the tracker. Try a different predictor (probably bad example)

predictor = UnscentedKalmanPredictor(target_transition_model)

tracker2 = PreBuiltSingleTargetTrackerNoClutter(detector=detector,
                                                ground_truth_prior=ground_truth_prior,
                                                predictor=predictor)

# See if this is any better ...
tracker2
