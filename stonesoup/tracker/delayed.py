import copy
import warnings
from datetime import timedelta, datetime

from stonesoup.base import Property
from stonesoup.feeder.customdetectionfeeder import CustomDetectionFeeder
from stonesoup.reader import DetectionReader
from stonesoup.tracker.base import Tracker
from stonesoup.types.detection import Detection
from stonesoup.writer import TrackWriter
from stonesoup.types.track import Track

import itertools
import operator
from typing import Tuple, Set, List


class FastForwardOldTracker(Tracker, TrackWriter):

    # Properties that should be edited
    tracker: Tracker = Property(doc="The 'current' tracker. This runs at the most up to date time")

    time_cut_off: timedelta = Property(
        default=timedelta(seconds=10),
        doc="Do not use detections after this time_cut_off")

    # These properties shouldn't be written to and are overwritten in the initialisation method
    detector: DetectionReader = Property(
        doc="This detector is used to provide detections to the tracker. It is not set, as it is "
            "taken from the 'tracker'.", default=None)
    delayed_tracker: Tracker = Property(default=None,
                                        doc="The delayed tracker will run 'time_cut_off' seconds "
                                            "behind the main tracker")
    detection_buffer: list = Property(default=None, doc="This stores")
    tracks_history: set = Property(default=None,
                                   doc="The historic output of the tracker is recorded here. The "
                                       "Track in self.tracker shows an edited history based off of"
                                       "newer detections than it had at the time. This records the "
                                       "output to measure the performance of the tracker")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if any(property_no_input is not None for property_no_input in
               (self.detector, self.delayed_tracker, self.detection_buffer, self.tracks_history)):
            warnings.warn("The detector, delayed_tracker, detection_buffer and tracks_history are "
                          "all overwritten in the initialisation method for the "
                          "FastForwardOldTracker class. The input values to these has been ignored")

        self.detector = self.tracker.detector
        self.tracker.detector = CustomDetectionFeeder()
        self.tracker.detector_iter = iter(self.tracker.detector)

        self.delayed_tracker = copy.deepcopy(self.tracker)
        self.detection_buffer = []

        self.iter_detector = iter(self.detector)
        self.tracks_history = set()

    def __next__(self) -> Tuple[datetime, Set[Track]]:

        tracker_time, a_detection_set = next(self.iter_detector)  # The time of the most recent detection that entered the tracker, set of detections
        last_detection_time_allowed = tracker_time - self.time_cut_off  # The last time of detection that can be used

        self.detection_buffer, old_detections = self.remove_older_detections(
            self.detection_buffer, last_detection_time_allowed)

        if len(old_detections) > 0:
            self.pass_old_detections_to_delayed_tracker(old_detections)

        new_detections = list(a_detection_set)

        if len(new_detections) > 0:

            new_detections, old_detections = self.remove_older_detections(
                new_detections, last_detection_time_allowed)
            if len(old_detections) > 0:
                warnings.warn('Detection is too old to accepted by filter')

            self.detection_buffer.extend(new_detections)

            if self.tracker.tracks == set():
                tracker_last_update_time = datetime.min
            else:
                tracker_last_update_time = max(track.state.timestamp for track in self.tracker.tracks)

            if all(tracker_last_update_time <= a_detection.timestamp for a_detection in new_detections):
                # All the detections are new
                track_output = self.pass_new_detections_to_current_tracker(new_detections)

            else:
                # Some of the detections are old
                self.tracker = copy.deepcopy(self.delayed_tracker)
                track_output = self.pass_all_detections_to_delayed_tracker()

        else:
            track_output = self.no_more_detections(tracker_time)

        self.record_history(track_output)

        return track_output

    def pass_old_detections_to_delayed_tracker(self, old_detections: List[Detection]):
        self.add_detections_to_tracker(old_detections, self.delayed_tracker)

    def pass_new_detections_to_current_tracker(self, new_detections: List[Detection])\
            -> Tuple[datetime, Set[Track]]:

        track_output = self.add_detections_to_tracker(new_detections, self.tracker)
        return track_output

    def pass_all_detections_to_delayed_tracker(self) -> Tuple[datetime, Set[Track]]:
        track_output = self.add_detections_to_tracker(self.detection_buffer, self.tracker)
        return track_output

    @property
    def tracks(self) -> Set[Track]:
        # return self.tracker.tracks
        return self.tracks_history

    def record_history(self, track_output: Tuple[datetime, Set[Track]]):
        """
        This records the state of the tracks at each timestep in self.tracks_history
        **This function is only implemented for a single target. ** It will raise
        'NotImplementedError' if there is more than one track produced by the tracker

        :param track_output: Current state of the tracker
        """
        time, tracks = track_output
        # self.tracks_history |= tracks

        if tracks == set():  # If empty, don't do anything
            return

        if len(tracks) > 1 or len(self.tracks_history) > 1:
            raise NotImplementedError("Current track recording is only valid for a single target")

        if self.tracks_history == set():
            self.tracks_history = copy.deepcopy(tracks)
        tracks_history = next(iter(self.tracks_history))
        current_track = next(iter(tracks))
        tracks_history.append(current_track.state)

    def no_more_detections(self, time: datetime) -> Tuple[datetime, Set[Track]]:
        blank_detections = [(time, set())]
        self.tracker.detector.available_detections = blank_detections
        track_output = next(self.tracker)
        return track_output

    def get_detections(self) -> List[Detection]:
        new_detections = []
        iter_detector = iter(self.detector)
        time, a_detection_set = next(iter_detector)
        if len(a_detection_set) is 0:
            a_detection = Detection(None, timestamp=time)
            new_detections.append(a_detection)
        else:
            new_detections.extend(a_detection_set)
        return new_detections

    @staticmethod
    def add_detections_to_tracker(detections: List[Detection],
                                  tracker: Tracker) -> Tuple[datetime, Set[Track]]:
        if len(detections) > 0:
            get_attr = operator.attrgetter('timestamp')
            grouped_detections = [list(g) for k, g in itertools.groupby(sorted(detections, key=get_attr), get_attr)]

            for detections in grouped_detections:
                detection_time = detections[0].timestamp
                output_detection = (detection_time, set(detections))
                tracker.detector.available_detections = [output_detection]
                track_output = next(tracker)
        else:
            track_output = next(tracker)

        return track_output

    @staticmethod
    def remove_older_detections(list_of_detections: List[Detection], time_limit: datetime)\
            -> Tuple[List[Detection], List[Detection]]:
        old_detections = []
        for a_detection in list_of_detections:
            if a_detection.timestamp < time_limit:
                old_detections.append(a_detection)
                list_of_detections.remove(a_detection)

        return list_of_detections, old_detections
