# -*- coding: utf-8 -*-
import datetime
from collections import defaultdict
from operator import attrgetter

import numpy as np
import scipy as sp
from scipy.spatial import KDTree
try:
    import rtree
except (ImportError, AttributeError) as err:
    import warnings
    warnings.warn(f"Failed to import 'rtree': {err!r}")
    rtree = None

from .base import DataAssociator
from ..base import Property
from ..models.base import LinearModel
from ..models.measurement import MeasurementModel
from ..predictor import Predictor
from ..types.update import Update
from ..updater import Updater


class DetectionKDTreeMixIn(DataAssociator):
    """Detection kd-tree based mixin

    Construct a kd-tree from detections and then use a :class:`~.Predictor` and
    :class:`~.Updater` to get prediction of track in measurement space. This is
    then queried against the kd-tree, and only matching detections are passed
    to the :attr:`hypothesiser`.

    Notes
    -----
    This is only suitable where measurements are in same space as each other
    and at the same timestamp.
    """
    predictor = Property(
        Predictor,
        doc="Predict tracks to detection times")
    updater = Property(
        Updater,
        doc="Updater used to get measurement prediction")
    number_of_neighbours = Property(
        int, default=None,
        doc="Number of neighbours to find. Default `None`, which means all "
            "points within the :attr:`max_distance` are returned.")
    max_distance = Property(
        float, default=np.inf,
        doc="Max distance to return points. Default `inf`")

    def generate_hypotheses(self, tracks, detections, timestamp, **kwargs):
        # No need for tree here.
        if not tracks:
            return {}
        if not detections:
            return {track: self.hypothesiser.hypothesise(
                track, detections, timestamp, **kwargs)
                for track in tracks}

        detections_list = list(detections)
        tree = KDTree(
            np.vstack([detection.state_vector[:, 0]
                       for detection in detections_list]))

        track_detections = defaultdict(set)
        for track in tracks:
            prediction = self.predictor.predict(track.state, timestamp)
            meas_pred = self.updater.predict_measurement(prediction)

            if self.number_of_neighbours is None:
                indexes = tree.query_ball_point(
                    meas_pred.state_vector.ravel(),
                    r=self.max_distance)
            else:
                _, indexes = tree.query(
                    meas_pred.state_vector.ravel(),
                    k=self.number_of_neighbours,
                    distance_upper_bound=self.max_distance)

            for index in np.atleast_1d(indexes):
                if index != len(detections_list):
                    track_detections[track].add(detections_list[index])

        return {track: self.hypothesiser.hypothesise(
            track, track_detections[track], timestamp, **kwargs)
            for track in tracks}


class TPRTreeMixIn(DataAssociator):
    """Detection TPR tree based mixin

    Construct a TPR-tree.
    """
    measurement_model = Property(
        MeasurementModel,
        doc="Measurement model used within the TPR tree")
    horizon_time = Property(
        datetime.timedelta,
        doc="How far the TPR tree should look into the future")
    vel_mapping = Property(
        np.ndarray,
        default=None,
        doc="Used to generate coordinates")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # if no vel_mapping take position mapping and plus 1 to each dimension
        # e.g. 0,2 would become 1,3
        self.pos_mapping = self.measurement_model.mapping
        if self.vel_mapping is None:
            self.vel_mapping = self.measurement_model.mapping[1::3]
        else:
            self.pos_mapping = self.measurement_model.mapping

        # Create tree
        tree_property = rtree.index.Property(
            type=rtree.index.RT_TPRTree,
            tpr_horizon=self.horizon_time.total_seconds(),
            dimension=len(self.pos_mapping))
        self._tree = rtree.index.RtreeContainer(properties=tree_property)
        self._coords = dict()

    def _track_tree_coordinates(self, track):
        state_vector = track.state_vector[self.pos_mapping, :]
        state_delta = 3 * np.sqrt(
            np.diag(track.covar)[self.pos_mapping].reshape(-1, 1))
        vel_vector = track.state_vector[self.vel_mapping, :]
        vel_delta = 3 * np.sqrt(
            np.diag(track.covar)[self.vel_mapping].reshape(-1, 1))

        min_pos = (state_vector - state_delta).ravel()
        max_pos = (state_vector + state_delta).ravel()
        min_vel = (vel_vector - vel_delta).ravel()
        max_vel = (vel_vector + vel_delta).ravel()

        return ((*min_pos, *max_pos), (*min_vel, *max_vel),
                track.timestamp.astimezone(datetime.timezone.utc).timestamp())

    def generate_hypotheses(self, tracks, detections, timestamp, **kwargs):
        # No need for tree here.
        if not tracks:
            return dict()

        c_time = None
        for track in sorted(tracks.union(self._tree),
                            key=attrgetter('timestamp')):
            if c_time is None:
                c_time = track.timestamp
            if track not in self._tree:
                self._coords[track] = self._track_tree_coordinates(track)
                self._tree.insert(track, self._coords[track])

            elif track not in tracks:
                coords = self._coords[track][:-1] \
                            + ((self._coords[track][-1] - 1e-6,
                                c_time.astimezone(datetime.timezone.utc).timestamp()),)
                self._tree.delete(track, coords)
                del self._coords[track]
            elif isinstance(track.state, Update):
                coords = self._coords[track][:-1] \
                            + ((self._coords[track][-1]-1e-6,
                                c_time.astimezone(datetime.timezone.utc).timestamp()),)
                self._tree.delete(track, coords)
                self._coords[track] = self._track_tree_coordinates(track)
                self._tree.insert(track, self._coords[track])
            c_time = track.timestamp

        track_detections = defaultdict(set)
        for detection in sorted(detections, key=attrgetter('timestamp')):
            if detection.measurement_model is not None:
                model = detection.measurement_model
            else:
                model = self.measurement_model

            if isinstance(model, LinearModel):
                model_matrix = model.matrix(**kwargs)
                inv_model_matrix = sp.linalg.pinv(model_matrix)
                state_meas = (inv_model_matrix
                              @ detection.state_vector)[self.pos_mapping, :]
            else:
                state_meas = model.inverse_function(
                    detection, **kwargs)[self.pos_mapping, :]

            det_time = detection.timestamp.astimezone(datetime.timezone.utc).timestamp()
            intersected_tracks = self._tree.intersection((
                (*state_meas.ravel(), *state_meas.ravel()),
                (0, 0)*len(self.pos_mapping),
                (det_time, det_time + 1e-3)))
            for track in intersected_tracks:
                track_detections[track].add(detection)

        return {track: self.hypothesiser.hypothesise(
            track, track_detections[track], timestamp, **kwargs)
            for track in tracks}
