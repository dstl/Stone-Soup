import copy
import datetime
from typing import List, Any, Tuple, Set
import itertools as it
from uuid import uuid4

import numpy as np
from matplotlib.path import Path
from scipy.stats import poisson
from shapely import unary_union

from reactive_isr_core.data import ImageStore, NetworkTopology, AssetList, RFI
from stonesoup.base import Base, Property
from stonesoup.custom.functions import eval_rfi_new
from stonesoup.custom.functions.rollout import CollectionAction, CommsAction, ProcAction, \
    rollout_actions, proc_actions_from_config_sequence, get_sensor, simulate_new_tracks
from stonesoup.custom.tracker import SMCPHD_JIPDA
from stonesoup.functions import gm_reduce_single
from stonesoup.tracker import Tracker
from stonesoup.types.array import StateVectors
from stonesoup.types.numeric import Probability
from stonesoup.types.state import ParticleState
from stonesoup.types.track import Track
from stonesoup.types.update import GaussianStateUpdate


class CommsAndProcEvaluator(Base):
    """A reward function which calculates the potential reduction in the uncertainty of track estimates
    if a particular action is taken by a sensor or group of sensors.

    Given a configuration of sensors and actions, a metric is calculated for the potential
    reduction in the uncertainty of the tracks that would occur if the sensing configuration
    were used to make an observation. A larger value indicates a greater reduction in
    uncertainty.
    """

    tracker: Tracker = Property(doc="Tracker used to track the tracks")
    num_timesteps: int = Property(doc="Number of timesteps to rollout")
    interval: datetime.timedelta = Property(doc="Interval between timesteps",
                                            default=datetime.timedelta(seconds=1))
    num_samples: int = Property(doc="Number of samples to take for each timestep", default=30)
    prob_survive: Probability = Property(doc="Probability of survival", default=Probability(0.99))
    use_variance: bool = Property(doc="Use variance in prioritisation", default=False)

    def __call__(self, config: Tuple[CollectionAction, CommsAction, ProcAction], tracks: Set[Track],
                 image_store: ImageStore, network_topology: NetworkTopology, assets: AssetList,
                 rfis: List[RFI], ongoing_actions, timestamp, *args, **kwargs):

        if not len(rfis):
            return 0

        # Rollout actions
        config_seq_list = rollout_actions(config, image_store, network_topology, assets, rfis,
                                          ongoing_actions, self.num_samples, self.num_timesteps,
                                          self.interval, timestamp)
        rewards = []
        # Evaluate each rollout
        for config_seq in config_seq_list:
            reward = 0
            # Get all processing actions
            proc_actions = proc_actions_from_config_sequence(config_seq)
            # Sort processing actions by image collection time
            sorted_proc_actions = sorted(proc_actions, key=lambda x: x.image.collection_time)
            tracks_copy = set(copy.copy(track) for track in tracks)

            # For each processing action
            for i, proc_action in enumerate(sorted_proc_actions):
                # Get the image and algorithm
                image = proc_action.image
                algorithm = proc_action.algorithm

                # The current time is the image collection time
                current_time = image.collection_time

                # Create a sensor
                sensor = get_sensor(image.location, image.fov_radius, algorithm.prob_detection,
                                    algorithm.false_alarm_density)

                # Predict tracks to current time
                predicted_tracks = set()
                for track in tracks_copy:
                    predicted_track = copy.copy(track)
                    predicted_track.append(self.tracker._predictor.predict(track, timestamp=current_time))
                    predicted_tracks.add(predicted_track)

                # Simulate new tracks
                new_tracks = simulate_new_tracks(sensor, current_time, self.tracker.birth_density)
                tracks_copy = set(tracks_copy)
                tracks_copy |= new_tracks
                predicted_tracks |= new_tracks

                # Use the sensor to generate detections
                detections = {detection
                              for detection in sensor.measure(predicted_tracks, noise=False,
                                                              timestamp=current_time)}
                # Configure the tracker's probability of detection based on the image footprint
                p = sensor.footprint
                self.tracker.prob_detect = _prob_detect_func([p],
                                                             proc_action.algorithm.prob_detection)
                self.tracker.clutter_intensity = proc_action.algorithm.false_alarm_density/p.area

                # Update tracks with detections
                tracks_copy = self._update_tracks(tracks_copy, detections, current_time)

                for rfi in rfis:
                    reward += eval_rfi_new(rfi, tracks_copy, use_variance=self.use_variance,
                                           timestamp=proc_action.end_time)
            rewards.append(reward)
        return np.max(rewards)

    def _update_tracks(self, tracks, detections, timestamp):
        tracks = list(tracks)
        hypotheses = self.tracker._associator.generate_hypotheses(tracks, detections, timestamp)
        associations = self.tracker._associator.associate(tracks, detections,
                                                          timestamp, hypotheses=hypotheses)
        for track, multihypothesis in associations.items():
            if isinstance(self.tracker, SMCPHD_JIPDA):
                # calculate each Track's state as a Gaussian Mixture of
                # its possible associations with each detection, then
                # reduce the Mixture to a single Gaussian State
                posterior_states = []
                posterior_state_weights = []
                for hypothesis in multihypothesis:
                    posterior_state_weights.append(hypothesis.probability)
                    if hypothesis:
                        posterior_states.append(self.tracker._updater.update(hypothesis))
                    else:
                        posterior_states.append(hypothesis.prediction)

                # Merge/Collapse to single Gaussian
                means = StateVectors([state.state_vector for state in posterior_states])
                covars = np.stack([state.covar for state in posterior_states], axis=2)
                weights = np.asarray(posterior_state_weights)

                post_mean, post_covar = gm_reduce_single(means, covars, weights)

                track.append(GaussianStateUpdate(
                    np.array(post_mean), np.array(post_covar),
                    multihypothesis,
                    multihypothesis[0].prediction.timestamp))
            else:
                timestamp_m1 = track.timestamp \
                    if self.tracker.predict else track[-2].timestamp
                time_interval = timestamp - timestamp_m1
                track.append(multihypothesis.prediction)
                prob_survive = np.exp(-self.tracker.prob_death * time_interval.total_seconds())
                pred_prob_exist = prob_survive * track.exist_prob
                non_exist_weight = 1 - pred_prob_exist
                target_hyps = hypotheses[track]
                if multihypothesis:
                    # Update track
                    state_post = self.tracker._updater.update(multihypothesis)
                    track.append(state_post)
                    weights = np.array([hyp.probability for hyp in target_hyps])*pred_prob_exist
                    new_exist_prob = np.sum(weights) / (non_exist_weight + np.sum(weights))
                    track.exist_prob = new_exist_prob
                else:
                    non_det_weight = target_hyps.get_missed_detection_probability()
                    new_exist_prob = non_det_weight / (non_exist_weight + non_det_weight)
                    track.exist_prob = new_exist_prob
        return tracks



def _prob_detect_func(fovs, prob_detect):
    """Closure to return the probability of detection function for a given environment scan"""
    # Get the union of all field of views
    fovs_union = unary_union(fovs)
    if fovs_union.geom_type == 'MultiPolygon':
        fovs = [poly for poly in fovs_union]
    else:
        fovs = [fovs_union]

    paths = [Path(poly.boundary.coords) for poly in fovs]

    # Probability of detection nested function
    def prob_detect_func(state):
        for path_p in paths:
            if isinstance(state, ParticleState):
                prob_detect_arr = np.full((len(state),), Probability(0.01))
                points = state.state_vector[[0, 2], :].T
                inside_points = path_p.contains_points(points)
                prob_detect_arr[inside_points] = prob_detect
                return prob_detect_arr
            else:
                points = state.state_vector[[0, 2], :].T
                return prob_detect if np.alltrue(path_p.contains_points(points)) \
                    else Probability(0.01)

    return prob_detect_func