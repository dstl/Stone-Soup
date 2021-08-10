import copy
import warnings
from datetime import timedelta, datetime

from stonesoup.base import Property
from stonesoup.feeder.customdetectionfeeder import CustomDetectionFeeder
from stonesoup.reader import DetectionReader
from stonesoup.tracker import Tracker
from stonesoup.tracker.base import Tracker
from stonesoup.types.detection import Detection
from stonesoup.writer import TrackWriter
from stonesoup.types.track import Track

import itertools
import operator


class FastForwardOldTracker(Tracker, TrackWriter):

    base_tracker: Tracker = Property(doc="The delayed tracker will run 'time_cut_off' seconds behind the main tracker")
    tracker: Tracker = Property(default=None)

    time_cut_off: timedelta = Property(
        default=timedelta(seconds=10),
        doc="Do not use detections after this time_cut_off")

    detector: DetectionReader = Property(default=None, doc="Detector used to generate detection objects.")
    delayed_tracker: Tracker = Property(default=None, doc="The delayed tracker will run 'time_cut_off' seconds behind "
                                                          "the main tracker")

    detection_buffer: list = Property(default=None, doc="Todo")

    latest_detection_time: datetime = Property(default=datetime.utcfromtimestamp(0),
                                               doc="The time of the most recent detection that entered the tracker")
    last_detection_time_allowed: datetime = Property(default=None, doc="The last time of detection that can be used")

    update_tracker: bool = Property(default=True, doc="Should the tracker be updated with empty detections")

    debug_tracker: bool = Property(default=False, doc="Should the tracker record detections that have passed through the tracker")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.detector = self.base_tracker.detector
        self.base_tracker.detector = CustomDetectionFeeder()
        self.base_tracker.detector_iter = iter(self.base_tracker.detector)

        self.delayed_tracker = copy.deepcopy(self.base_tracker)
        self.detection_buffer = []

        self.last_detection_time_allowed = self.latest_detection_time - self.time_cut_off
        self.tracker=self.base_tracker
        self.iter_detector = iter(self.detector)
        self.tracks_history = set()

        if self.debug_tracker:
            from collections import OrderedDict
            self.events = []

    def __next__(self):
        self.detection_buffer, old_detections = self.remove_older_detections(
            self.detection_buffer, self.last_detection_time_allowed)
        if len(old_detections) > 0:

            if self.debug_tracker:
                print("Add Detection to Delayed Tracker:")
                self.events.append((['Old detections added to delayed tracker'], copy.copy(old_detections)))
            self.add_detections_to_tracker(old_detections, self.delayed_tracker)


        time, a_detection_set = next(self.iter_detector)

        new_detections = list(a_detection_set)

        if len(new_detections) > 0:

            new_detections, old_detections = self.remove_older_detections(
                new_detections, self.last_detection_time_allowed)
            if len(old_detections) > 0:
                warnings.warn('Detection is too old to accepted by filter')

            self.detection_buffer.extend(new_detections)

            if all(self.latest_detection_time <= a_detection.timestamp for a_detection in new_detections):
                # All the detections are new

                if self.debug_tracker:
                    print("Adding New Detections to Tracker:")
                    self.events.append((['New Detections added to new tracker'], copy.copy(new_detections)))

                track_output = self.add_detections_to_tracker(new_detections, self.base_tracker)
            else:
                # Some of the detections are old

                if self.debug_tracker:
                    print("Updating old tracker with buffer:")
                    self.events.append((['Old tracker copied to new tracker. Detection buffer added to new (was old) tracker'],copy.copy(self.detection_buffer)))
                self.base_tracker = copy.deepcopy(self.delayed_tracker)
                track_output = self.add_detections_to_tracker(self.detection_buffer, self.base_tracker)

            self.update_timestamps()
            # self.latest_detection_time, _ = track_output

        else:
            track_output = self.no_more_detections()

            if self.update_tracker:
                self.latest_detection_time = time

        self.record_history(track_output)

        return track_output

    @property
    def tracks(self):
        # return self.base_tracker.tracks
        return self.tracks_history

    def record_history(self, track_output):
        time, tracks = track_output
        # self.tracks_history |= tracks

        if tracks == set():  # If empty, don't do anything
            return

        if len(tracks) > 1 or len(self.tracks_history) > 1:
            raise NotImplementedError("Current track recording is only valid for a single target")

        if self.tracks_history == set():
            self.tracks_history = copy.deepcopy(tracks)
        track_history = next(iter(self.tracks_history))
        current_track = next(iter(tracks))
        track_history.append(current_track.state)

    def no_more_detections(self):
        blank_detections = [(self.latest_detection_time, set())]
        self.base_tracker.detector.detections = blank_detections
        track_output = next(self.base_tracker)
        return track_output

    def update_timestamps(self):
        self.latest_detection_time = max(track.state.timestamp for track in self.base_tracker.tracks)
        self.last_detection_time_allowed = self.latest_detection_time - self.time_cut_off

    def get_detections(self):
        new_detections = []
        iter_detector = iter(self.detector)
        time, a_detection_set = next(iter_detector)
        if len(a_detection_set) is 0:
            a_detection = Detection(None, timestamp=time)
            new_detections.append(a_detection)
        else:
            new_detections.extend(a_detection_set)
        return new_detections

    def get_all_tracks(self):
        states = []
        for _, tracks in self:
            for track in tracks:
                states.append(track.state)
        track.states = states
        return track

    def get_all_tracks2(self):

        states = []
        # infinite loop
        while True:
            try:
                # get the next item
                _, tracks = next(self)
                for track in tracks:
                    states.append(track.state)
                # do something with element
            except StopIteration:
                # if StopIteration is raised, break from loop
                break

        _, tracks = next(self)
        track.states = states
        return track

    @staticmethod
    def add_detections_to_tracker(detections, tracker):
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
    def remove_older_detections(list_of_detections, time_limit):
        old_detections = []
        for a_detection in list_of_detections:
            if a_detection.timestamp < time_limit:
                old_detections.append(a_detection)
                list_of_detections.remove(a_detection)

        return list_of_detections, old_detections

"""
from stonesoup.base import Base
class EventRecorder(Base):
    record
    events: OrderedDict = Property(doc="The delayed tracker will run 'time_cut_off' seconds behind the main tracker")
"""