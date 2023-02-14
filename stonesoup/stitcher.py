from collections import defaultdict
from copy import deepcopy
from datetime import timedelta
from functools import lru_cache

import numpy as np
from scipy.optimize import linear_sum_assignment

from .base import Base, Property
from .hypothesiser import Hypothesiser
from .models.measurement.linear import LinearGaussian
from .types.detection import Detection
from .types.track import Track


class TrackStitcher(Base):
    forward_hypothesiser: Hypothesiser = Property(
        doc="Forward predicting hypothesiser.", default=None)
    backward_hypothesiser: Hypothesiser = Property(
        doc="Backward predicting hypothesiser.", default=None)

    @staticmethod
    @lru_cache()
    def _extract_detection(track, backward=False):
        state = track[0] if not backward else track[-1]
        return Detection(state_vector=state.state_vector,
                         timestamp=state.timestamp,
                         measurement_model=LinearGaussian(
                             ndim_state=track.ndim,
                             mapping=list(range(track.ndim)),
                             noise_covar=state.covar),
                         metadata=track.id)

    @staticmethod
    def _get_track(track_id, tracks):
        for track in tracks:
            if track.id == track_id:
                return track

    @staticmethod
    def _merge(a, b):
        if a[-1] == b[0]:
            a.pop(-1)
            return a + b
        else:
            return a

    def forward_predict(self, tracks, start_time):
        x_forward = defaultdict(dict)
        for n in range(int((min(track[0].timestamp for track in tracks) -
                            start_time).total_seconds()),
                       int((max(track[-1].timestamp for track in tracks) -
                            start_time).total_seconds())):
            poss_tracks = []
            poss_detections = set()
            for track in tracks:
                timestamp = start_time + timedelta(seconds=n)
                if track[-1].timestamp < timestamp:
                    poss_tracks.append(track)
                if track[0].timestamp == timestamp:
                    poss_detections.add(self._extract_detection(track))
            if not poss_detections:
                continue
            for track in poss_tracks:
                hypotheses = self.forward_hypothesiser.hypothesise(
                    track, poss_detections, timestamp)
                for hypothesis in hypotheses:
                    if hypothesis:
                        x_forward[track.id][hypothesis.measurement.metadata] = hypothesis.distance
                    else:
                        x_forward[track.id][None] = hypothesis.distance
        return x_forward

    def backward_predict(self, tracks, start_time):
        x_backward = defaultdict(dict)
        for n in range(int((max(track[-1].timestamp for track in tracks) -
                            start_time).total_seconds()),
                       int((min(track[0].timestamp for track in tracks) -
                            start_time).total_seconds()),
                       -1):
            poss_tracks = []
            poss_detections = set()
            for track in tracks:
                timestamp = start_time + timedelta(seconds=n)
                if track[0].timestamp > timestamp:
                    poss_tracks.append(track)
                if track[-1].timestamp == timestamp:
                    poss_detections.add(self._extract_detection(track, backward=True))
                if not poss_detections:
                    continue
                for track in poss_tracks:
                    hypotheses = self.backward_hypothesiser.hypothesise(
                        track, poss_detections, start_time + timedelta(seconds=n))
                    missed_hyp = next(hypothesis for hypothesis in hypotheses if not hypothesis)
                    for hypothesis in hypotheses:
                        if hypothesis:
                            x_backward[hypothesis.measurement.metadata][track.id] =\
                                hypothesis.distance
                            # TODO: Not ideal. Is there a better way?
                            x_backward[hypothesis.measurement.metadata][None] = missed_hyp.distance
        return x_backward

    @staticmethod
    def _merge_forward_and_backward(x_forward, x_backward):
        x = defaultdict(dict)
        for key in x_forward.keys() | x_backward.keys():
            if key not in x_forward and key in x_backward:
                x[key] = x_backward[key]
            elif key in x_forward[key] and key not in x_backward:
                x[key] = x_forward[key]
            else:
                arr = dict()
                missed_f_val = missed_b_val = None

                for f_id, f_val in x_forward[key].items():
                    if f_id is None:
                        missed_f_val = f_val  # May be needed later
                    for b_id, b_val in x_backward[key].items():
                        if b_id is None:
                            missed_b_val = b_val  # May be needed later
                        if f_id == b_id:
                            arr[f_id] = f_val + b_val
                for f_id, f_val in x_forward[key].items():
                    if f_id not in arr:
                        if missed_b_val is None:
                            raise RuntimeError("Missing distance for backward during merge")
                        arr[f_id] = f_val + missed_b_val
                for b_id, b_val in x_backward[key].items():
                    if b_id not in arr:
                        if missed_f_val is None:
                            raise RuntimeError("Missing distance for forward during merge")
                        arr[b_id] = b_val + missed_f_val
                x[key] = arr
            return x

    def stitch(self, tracks, start_time):
        forward, backward = False, False
        if self.forward_hypothesiser is not None:
            forward = True
        if self.backward_hypothesiser is not None:
            backward = True

        if forward:
            x_forward = self.forward_predict(tracks, start_time)
        if backward:
            x_backward = self.backward_predict(tracks, start_time)

        if forward and not backward:
            x = x_forward
        elif not forward and backward:
            x = x_backward
        else:
            x = self._merge_forward_and_backward(x_forward, x_backward)

        i_track_ids = set(x.keys())
        j_track_ids = {id_ for combo in x.values() for id_ in combo if id_ is not None}
        j_track_ids |= i_track_ids  # Space for missed hypotheses

        matrix_val = np.full((len(i_track_ids), len(j_track_ids)), np.inf)
        matrix_track = [[(i_track_id, None)] * len(j_track_ids) for i_track_id in i_track_ids]

        for i, i_track_id in enumerate(i_track_ids):
            for j, j_track_id in enumerate(j_track_ids):
                if j_track_id in x[i_track_id]:
                    matrix_val[i][j] = x[i_track_id][j_track_id]
                    matrix_track[i][j] = (i_track_id, j_track_id)
                elif None in x[i_track_id]:
                    matrix_val[i][j] = x[i_track_id][None]

        row_ind, col_ind = linear_sum_assignment(matrix_val)

        for i in range(len(col_ind)):
            start_track, end_track = matrix_track[row_ind[i]][col_ind[i]]
            x[start_track] = end_track

        combo = []
        for key in x:
            if x[key] is None:
                continue
            elif len(combo) == 0 or not (
                    any(key in sublist for sublist in combo) or
                    any(x[key] in sublist for sublist in combo)):
                combo.append([key, x[key]])
            elif any(x[key] in sublist for sublist in combo):
                for track_list in combo:
                    if x[key] in track_list:
                        track_list.insert(track_list.index(x[key]), key)
            else:
                for track_list in combo:
                    if key in track_list:
                        track_list.insert(track_list.index(key) + 1, x[key])

        i = 0
        count = 0
        while i != len(combo):
            id1 = combo[i]
            id2 = combo[count]
            new_list1 = self._merge(deepcopy(id1), deepcopy(id2))
            new_list2 = self._merge(deepcopy(id2), deepcopy(id1))
            if len(new_list1) == len(id1) and len(new_list2) == len(id2):
                count += 1
            else:
                combo.remove(id1)
                combo.remove(id2)
                count = 0
                i = 0
                if len(new_list1) > len(id1):
                    combo.append(new_list1)
                else:
                    combo.append(new_list2)
            if count == len(combo):
                count = 0
                i += 1
                continue

        tracks = set(tracks)
        for ids in combo:
            x = []
            for a in ids:
                track = self._get_track(a, tracks)
                x = x + track.states
                tracks.remove(track)
            tracks.add(Track(x))

        return tracks
