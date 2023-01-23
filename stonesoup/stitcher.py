from datetime import timedelta
from stonesoup.types.track import Track
from stonesoup.types.detection import Detection, TrueDetection
from stonesoup.models.measurement.linear import LinearGaussian
from copy import deepcopy
from scipy.optimize import linear_sum_assignment


class TrackStitcher():
    def __init__(self, forward_hypothesiser=None, backward_hypothesiser=None):
        self.forward_hypothesiser = forward_hypothesiser
        self.backward_hypothesiser = backward_hypothesiser

    @staticmethod
    def __extract_detection(track, backward=False):
        state = track[0]
        if backward:
            state = track[-1]
        return Detection(state_vector=state.state_vector,
                         timestamp=state.timestamp,
                         measurement_model=LinearGaussian(
                             ndim_state=2 * n_spacial_dimensions,
                             mapping=list(range(2 * n_spacial_dimensions)),
                             noise_covar=state.covar),
                         metadata=track.id)

    @staticmethod
    def __get_track(track_id, tracks):
        for track in tracks:
            if track.id == track_id:
                return track

    @staticmethod
    def __merge(a, b):
        if a[-1] == b[0]:
            a.pop(-1)
            return a + b
        else:
            return a

    def forward_predict(self, tracks, start_time):
        x_forward = {track.id: [] for track in tracks}
        poss_pairs = []
        for n in range(int((min([track[0].timestamp for track in tracks]) - start_time).total_seconds()),
                       int((max([track[-1].timestamp for track in tracks]) - start_time).total_seconds())):
            poss_tracks = []
            poss_detections = set()
            for track in tracks:
                if track[-1].timestamp < start_time + timedelta(seconds=n):
                    poss_tracks.append(track)
                if track[0].timestamp == start_time + timedelta(seconds=n):
                    poss_detections.add(self.__extract_detection(track))
            if len(poss_tracks) > 0 and len(poss_detections) > 0:
                for track in poss_tracks:
                    a = self.forward_hypothesiser.hypothesise(track, poss_detections, start_time + timedelta(seconds=n))
                    if a[0].measurement.metadata == {}:
                        continue
                    else:
                        x_forward[track.id].append((a[0].measurement.metadata, a[0].distance))
        return x_forward

    def backward_predict(self, tracks, start_time):
        x_backward = {track.id: [] for track in tracks}
        poss_pairs = []
        for n in range(int((max([track[-1].timestamp for track in tracks]) - start_time).total_seconds()),
                       int((min([track[0].timestamp for track in tracks]) - start_time).total_seconds()),
                       -1):
            poss_tracks = []
            poss_detections = set()
            for track in tracks:
                if track[0].timestamp > start_time + timedelta(seconds=n):
                    poss_tracks.append(track)
                if track[-1].timestamp == start_time + timedelta(seconds=n):
                    poss_detections.add(self.__extract_detection(track, backward=True))
            if len(poss_tracks) > 0 and len(poss_detections) > 0:
                for track in poss_tracks:
                    a = self.backward_hypothesiser.hypothesise(track, poss_detections,
                                                               start_time + timedelta(seconds=n))
                    if a[0].measurement.metadata == {}:
                        continue
                    else:
                        x_backward[a[0].measurement.metadata].append((track.id, a[0].distance))
        return x_backward

    def stitch(self, tracks, start_time):
        forward, backward = False, False
        if self.forward_hypothesiser != None:
            forward = True
        if self.backward_hypothesiser != None:
            backward = True

        tracks = list(tracks)
        x = {track.id: [] for track in tracks}
        if forward:
            x_forward = self.forward_predict(tracks, start_time)
        if backward:
            x_backward = self.backward_predict(tracks, start_time)

        if forward and not (backward):
            x = x_forward
        elif not (forward) and backward:
            x = x_backward
        else:
            for key in x:
                if x_forward[key] == [] and x_backward[key] == []:
                    x[key] = []
                elif x_forward[key] == [] and x_backward[key] != []:
                    x[key] = x_backward[key]
                elif x_forward[key] != [] and x_backward[key] == []:
                    x[key] = x_forward[key]
                else:
                    arr = []
                    for f_val in x_forward[key]:
                        for b_val in x_backward[key]:
                            if f_val[0] == b_val[0]:
                                arr.append((f_val[0], f_val[1] + b_val[1]))
                    for f_val in x_forward[key]:
                        in_arr = False
                        for a_val in arr:
                            if f_val[0] == a_val[0]:
                                in_arr = True
                        if not (in_arr):
                            arr.append((f_val[0], f_val[1] + 300))
                    for b_val in x_backward[key]:
                        in_arr = False
                        for a_val in arr:
                            if b_val[0] == a_val[0]:
                                in_arr = True
                        if not (in_arr):
                            arr.append((b_val[0], b_val[1] + 300))
                    x[key] = arr

        matrix_val = [[300] * len(tracks) for i in range(len(tracks))]
        matrix_track = [[None] * len(tracks) for i in range(len(tracks))]
        for i in range(len(tracks)):
            for j in range(len(tracks)):
                if tracks[i].id in x:
                    if tracks[j].id in [combo[0] for combo in x[tracks[i].id]]:
                        matrix_val[i][j] = [tup[1] for tup in x[tracks[i].id] if tup[0] == tracks[j].id][0]
                        matrix_track[i][j] = (tracks[i].id, tracks[j].id)
                    else:
                        matrix_track[i][j] = (tracks[i].id, None)

        row_ind, col_ind = linear_sum_assignment(matrix_val)

        for i in range(len(col_ind)):
            start_track = matrix_track[row_ind[i]][col_ind[i]][0]
            end_track = matrix_track[row_ind[i]][col_ind[i]][1]
            if end_track == None:
                x[start_track] = None
            else:
                x[start_track] = end_track

        combo = []
        for key in x:
            if x[key] is None:
                continue
            elif len(combo) == 0 or not (
                    any(key in sublist for sublist in combo) or any(x[key] in sublist for sublist in combo)):
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
            new_list1 = self.__merge(deepcopy(id1), deepcopy(id2))
            new_list2 = self.__merge(deepcopy(id2), deepcopy(id1))
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
                track = self.__get_track(a, tracks)
                x = x + track.states
                tracks.remove(track)
            tracks.add(Track(x))

        return tracks
