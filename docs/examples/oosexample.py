import copy
from typing import List, Tuple, Set

from stonesoup.tracker.delayed import FastForwardOldTracker
from stonesoup.types.detection import Detection
from stonesoup.types.state import State, GaussianState
from stonesoup.feeder.base import DetectionFeeder
from stonesoup.feeder.simple import SimpleDetectionFeeder
from stonesoup.base import Property, Base
from stonesoup.buffered_generator import BufferedGenerator
from stonesoup.tracker.prebuilttrackers import PreBuiltSingleTargetTrackerNoClutter
from stonesoup.tracker.simple import SingleTargetTracker as _SingleTargetTracker
from stonesoup.models.transition.linear import (
    ConstantVelocity,
    CombinedLinearGaussianTransitionModel
)
import numpy as np
from datetime import datetime, timedelta
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.types.track import Track


class SimpleDetectionFeederWithComments(SimpleDetectionFeeder):
    """
    This class is exactly the same as SimpleDetectionFeeder aside from there being additional print
    comments to aid debugging and demonstration purposes
    """
    @BufferedGenerator.generator_method
    def data_gen(self):
        for detection_time, set_of_detections in super().data_gen():
            for detection in set_of_detections:
                print('release detection', detection.metadata['id'])
                if detection.metadata['id'] == 13:
                    five = 5
            yield detection_time, set_of_detections


class SimpleDetectionFeederWithComments2(DetectionFeeder):

    #reader = Property(list, default=None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.det_iter = iter(self.reader)

    @BufferedGenerator.generator_method
    def data_gen(self):
        for detection in self.reader:
            print("")
            print('release detection', detection.metadata['id'])
            if detection.metadata['id'] == 13:
                five = 5
            yield detection.timestamp, {detection}


class SingleTargetTracker(_SingleTargetTracker):
    """
    This class is exactly the same as SingleTargetTracker in stonesoup.tracker.simple aside from
    there being additional print comments to aid debugging and demonstration purposes
    """
    def __next__(self):
        time, detections = next(self.detector_iter)
        print("Tracker processing", time, [detection.metadata['id'] for detection in detections],
              [detection.metadata['Origin'] for detection in detections], "time/id/origin")
        if self._track is not None:
            associations = self.data_associator.associate(
                self.tracks, detections, time)
            if associations[self._track]:
                state_post = self.updater.update(associations[self._track])
                self._track.append(state_post)
            else:
                self._track.append(
                    associations[self._track].prediction)

        if self._track is None or self.deleter.delete_tracks(self.tracks):
            new_tracks = self.initiator.initiate(detections, time)
            if new_tracks:
                self._track = new_tracks.pop()
            else:
                self._track = None

        return time, self.tracks


class FastForwardOldTrackerDebugged(FastForwardOldTracker):
    """
    This class is exactly the same as FastForwardOldTracker in stonesoup.tracker.delayed aside from
    there being additional print comments to aid debugging and demonstration purposes and a method
    of recording events
    """
    debug_tracker: bool = Property(default=True,
                                   doc="Should the tracker record detections that have passed through the tracker")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.debug_tracker:
            self.events = []

    def pass_old_detections_to_delayed_tracker(self, old_detections: List[Detection]):
        if self.debug_tracker:
            print("Add Detection to Delayed Tracker:")
            self.events.append(
                (['Old detections added to delayed tracker'], copy.copy(old_detections)))
        super().add_detections_to_tracker(old_detections, self.delayed_tracker)

    def pass_new_detections_to_current_tracker(self, new_detections: List[Detection]) \
            -> Tuple[datetime, Set[Track]]:

        if self.debug_tracker:
            print("Adding New Detections to Tracker:")
            self.events.append((['New Detections added to new tracker'], copy.copy(new_detections)))

        return super().add_detections_to_tracker(new_detections, self.tracker)

    def pass_all_detections_to_delayed_tracker(self) -> Tuple[datetime, Set[Track]]:
        if self.debug_tracker:
            print("Updating old tracker with buffer:")
            self.events.append(([
                                    'Old tracker copied to new tracker. Detection buffer added to new (was old) tracker'],
                                copy.copy(self.detection_buffer)))

        return super().add_detections_to_tracker(self.detection_buffer, self.tracker)


'''
The scenario is that we have two sensors producing detections for a tracker. Sensor 1 is close to the tracker and can 
supply detections immediately. Sensor is remote and further away the tracker. There is a delay in the detections from Sensor 2 
reaching the tracker. The delay in the detections is given by the ‘time_delay’ variable.
'''
start_time = datetime.now()

time_delay = 5  # seconds

seconds_in_sim = 20


# The target is moving from the origin (0,0,0) north towards (0,x,0) at one unit per second.
def get_target_position(target_time):
    time_since_start = target_time - start_time
    return [0, time_since_start.total_seconds(), 0]


def get_target_position2(target_time, std):
    target_position = get_target_position(target_time)
    target_position[0] = np.random.randn()*std
    return target_position


measurement_model = LinearGaussian(
    ndim_state=6,  # Number of state dimensions (position and velocity in 3D)
    mapping=(0, 2, 4),  # Mapping measurement vector index to state index
    noise_covar=np.diag([15]*3)  # Covariance matrix for Gaussian PDF
    )

sensor_1_offset = [-1, 0, 0]
sensor_2_offset = [+1, 0, 0]

sensor_1_detections = []
sensor_2_detections = []

detection_x_standard_dev = 0.2

for idx, time in enumerate([start_time + timedelta(seconds=x/2) for x in range(2*seconds_in_sim)]):
    target_pos = get_target_position2(time, detection_x_standard_dev)

    if idx % 2 == 0:
        sensor_1_detections.append(Detection(state_vector=[x+y for x, y in zip(target_pos, sensor_1_offset)],
                                             timestamp=time,
                                             measurement_model=measurement_model,
                                             metadata={'Origin': 1,
                                                       'id': idx*2,
                                                       'time_at_tracker': time}))
    else:
        sensor_2_detections.append(Detection(state_vector=[x+y for x, y in zip(target_pos, sensor_2_offset)],
                                             timestamp=time,  # + timedelta(seconds=0.5),
                                             measurement_model=measurement_model,
                                             metadata={'Origin': 2,
                                                       'id': (idx*2)+1,
                                                       'time_at_tracker': time + timedelta(seconds=time_delay)}))


all_detections = [*sensor_1_detections, *sensor_2_detections]


motion_model_noise = 0.001
target_transition_model = CombinedLinearGaussianTransitionModel(
    (ConstantVelocity(motion_model_noise), ConstantVelocity(motion_model_noise),
     ConstantVelocity(motion_model_noise)))
ground_truth_prior=GaussianState(state_vector=[0,0,0,0,0,0],timestamp=start_time,covar=np.diag([1]*6))

from stonesoup.plotter import Plotter
plotter = Plotter()
from matplotlib import pyplot as plt
plotter.plot_measurements(all_detections, [0, 2])
plotter.fig

if False:
    tracker_template = PreBuiltSingleTargetTrackerNoClutter(detector=SimpleDetectionFeederrWithComments(reader=all_detections),
                                                            ground_truth_prior=ground_truth_prior,
                                                            target_transition_model=target_transition_model)
    tracker = tracker_template.tracker
    for time, track in tracker:
        pass




    plotter.plot_tracks(tracker.tracks, [0, 2], uncertainty=True)
    plotter.fig


    #plt.draw()
    #plt.show()



detections_reordered = list(all_detections)

detections_reordered = sorted(detections_reordered, key=lambda x: x.metadata['time_at_tracker'])



tracker_template = PreBuiltSingleTargetTrackerNoClutter(detector=SimpleDetectionFeederWithComments(reader=detections_reordered),
                                                        ground_truth_prior=ground_truth_prior,
                                                        target_transition_model=target_transition_model)
standard_tracker = SingleTargetTracker(**tracker_template.get_kwargs())
tracker2 = FastForwardOldTrackerDebugged(tracker=standard_tracker, time_cut_off=timedelta(seconds=5))

for time, track in tracker2:
    tracker_show = tracker2.tracker
    time_label = "time=" + str((time-start_time).total_seconds()) + "s"
    if len(tracker_show.tracks) > 0 and False:
        plotter.plot_tracks(tracker_show.tracks, [0, 2], uncertainty=False, track_label="New Track at "+time_label)
        plotter.fig

        plt.show(block=False)
        plt.pause(0.2)

    tracker_show = tracker2.delayed_tracker
    if len(tracker_show.tracks) > 0 and False:
        plotter.plot_tracks(tracker_show.tracks, [0, 2], uncertainty=False, track_label="Delayed Track at "+time_label)
        plotter.fig

        plt.show(block=False)
        plt.pause(0.2)

    tracker_show = tracker2.tracker
    if len(tracker_show.tracks) > 0 and True:
        plotter.plot_tracks(track, [0, 2], uncertainty=False, track_label="Current Track at "+time_label)
        plotter.fig

        plt.show(block=False)
        plt.pause(0.2)


plotter.plot_tracks(tracker2.tracks_history, [0, 2], uncertainty=False, track_label="Track History")
plotter.fig

plt.draw()
plt.show()


five=5