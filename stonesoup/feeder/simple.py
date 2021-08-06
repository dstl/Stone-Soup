import copy
import datetime
from abc import abstractmethod
from typing import Iterable, Callable

from stonesoup.base import Property
from stonesoup.buffered_generator import BufferedGenerator
from stonesoup.feeder import Feeder
from stonesoup.feeder.base import DetectionFeeder
from stonesoup.types.detection import Detection


class SimpleFeeder(Feeder):

    reader: Iterable = Property(doc="Source of states")

    @BufferedGenerator.generator_method
    def data_gen(self):
        for item in self.reader:
            yield item


class SimpleDetectionFeeder(SimpleFeeder, DetectionFeeder):
    reader: Iterable[Detection] = Property(doc="Source of detections")

    @BufferedGenerator.generator_method
    def data_gen(self):
        for item in self.reader:
            yield item.timestamp, {item}


class IterDetectionFeeder(DetectionFeeder):

    def __iter__(self):
        return self

    @abstractmethod
    def __next__(self):
        ...

    @BufferedGenerator.generator_method
    def data_gen(self):
        detection_iter = iter(self)
        for time, detections in detection_iter:
            yield (time, detections)


class TimeSteppedFeeder(IterDetectionFeeder):

    time: datetime.datetime = Property(default=None,
                                       doc="If none, find earliest time of detections")
    time_step: datetime.timedelta = Property(
        default=datetime.timedelta(seconds=1),
        doc="Time window to group detections")

    detection_buffer: Iterable[Detection] = Property(default=None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.detection_buffer is None:
            self.detection_buffer = []
            self.read_all_detections()
        else:
            self.detection_buffer = [*self.detection_buffer]  # Shallow copy detections into a list

        if self.time is None:
            self.get_start_time()

    def read_all_detections(self):
        for time, set_of_detections in self.reader:
            self.detection_buffer.extend(set_of_detections)

    def get_start_time(self):
        self.time = min(state.timestamp for state in self.detection_buffer)

    def __next__(self):
        if len(self.detection_buffer) is 0:
            raise StopIteration

        min_time = self.time
        max_time = self.time + self.time_step
        self.time = max_time

        detections_to_release, detections_to_keep = [], []
        for detection in self.detection_buffer:
            if min_time <= detection.timestamp < max_time:
                detections_to_release.append(detection)
            else:
                detections_to_keep.append(detection)

        self.detection_buffer = detections_to_keep

        return min_time, set(detections_to_release)

    def __copy__(self):
        """
        The default copy function doesn't copy the properties, it points to the same objects. This
        overrides the default copy function and makes a shallow copy of all of the properties in the
        class.
        """
        cls = self.__class__
        result = cls.__new__(cls)
        new_dict = {k: copy.copy(v) for k, v in self.__dict__.items()}
        result.__dict__.update(new_dict)
        return result


class ConditionalTimeSteppedFeeder(TimeSteppedFeeder):

    condition: Callable[[Detection, datetime.datetime, datetime.datetime], bool] = Property(
        default=lambda detection, min_time, max_time: min_time <= detection.timestamp < max_time,
        doc="Must include a function in this form"
    )
    end_condition: Callable[[Iterable[Detection]], bool] = Property(
        default=None,
        doc="A condition to stop iterating. When no more valid detections are present")

    def __next__(self):
        if len(self.detection_buffer) is 0:
            raise StopIteration
        if self.end_condition is not None:
            if self.end_condition(self.detection_buffer):
                raise StopIteration

        min_time = self.time
        max_time = self.time + self.time_step
        self.time = max_time

        detections_to_release, detections_to_keep = [], []
        for detection in self.detection_buffer:
            if self.condition(detection, min_time, max_time):
                detections_to_release.append(detection)
            else:
                detections_to_keep.append(detection)

        self.detection_buffer = detections_to_keep

        return min_time, set(detections_to_release)


def tes_detections():
    time = datetime.datetime(2019, 4, 1, 14)
    time_step = datetime.timedelta(seconds=1)

    yield time, {
        Detection([[50], [0]], timestamp=time,
                  metadata={'colour': 'red',
                            'score': 0}),
        Detection([[20], [5]], timestamp=time,
                  metadata={'colour': 'green',
                            'score': 0.5}),
        Detection([[1], [1]], timestamp=time,
                  metadata={'colour': 'blue',
                            'score': 0.1}),
    }

    time += time_step
    yield time, {
        Detection([[-5], [4]], timestamp=time,
                  metadata={'colour': 'red',
                            'score': 0.4}),
        Detection([[11], [200]], timestamp=time,
                  metadata={'colour': 'green'}),
        Detection([[0], [0]], timestamp=time,
                  metadata={'colour': 'green',
                            'score': 0.2}),
        Detection([[-43], [-10]], timestamp=time,
                  metadata={'colour': 'blue',
                            'score': 0.326}),
    }

    time += time_step
    yield time, {
        Detection([[561], [10]], timestamp=time,
                  metadata={'colour': 'red',
                            'score': 0.745}),
        Detection([[1], [-10]], timestamp=time - time_step / 2,
                  metadata={'colour': 'red',
                            'score': 0}),
        Detection([[-11], [-50]], timestamp=time,
                  metadata={'colour': 'blue',
                            'score': 2}),
    }

    time += time_step
    yield time, {
        Detection([[1], [-5]], timestamp=time,
                  metadata={'colour': 'red',
                            'score': 0.3412}),
        Detection([[1], [-5]], timestamp=time,
                  metadata={'colour': 'blue',
                            'score': 0.214}),
    }

    time += time_step
    yield time, {
        Detection([[-11], [5]], timestamp=time,
                  metadata={'colour': 'red',
                            'score': 0.5}),
        Detection([[13], [654]], timestamp=time,
                  metadata={'colour': 'blue',
                            'score': 0}),
        Detection([[-3], [6]], timestamp=time,
                  metadata={}),
    }


def tes_1():
    feeder = TimeSteppedFeeder(reader=tes_detections())
    feeder_true = tes_detections()

    for (time1, dets1), (time2, dets2) in zip(feeder, feeder_true):
        x = {str(det) for det in dets1}
        y = {str(det) for det in dets2}
        print("Difference: ", x ^ y)
        print("------------------------------------------")

def tes_2():
    rule = lambda detection, min_time, max_time: (min_time <= detection.timestamp < max_time) and detection.metadata.get('colour') == 'red'
    other_rule = lambda detections: not any(detection.metadata.get('colour') == 'red' for detection in detections)
    feeder = ConditionalTimeSteppedFeeder(reader=tes_detections(),
                                          condition=rule,
                                          end_condition=other_rule)
    feeder_true = tes_detections()

    for (time1, dets1), (time2, dets2) in zip(feeder, feeder_true):
        x = {str(det) for det in dets1}
        y = {str(det) for det in dets2}
        print("Difference: ", *(x ^ y))
        print("------------------------------------------")


def tes_3():
    rule = lambda detection, min_time, max_time: (min_time <= detection.timestamp < max_time) and detection.metadata.get('colour') == 'red'
    other_rule = lambda detections: not any(detection.metadata.get('colour') == 'red' for detection in detections)
    feeder = ConditionalTimeSteppedFeeder(reader=tes_detections(),
                                          condition=rule,
                                          end_condition=other_rule)

    all_dets = set()
    for time1, dets1 in feeder:
        all_dets = all_dets | dets1

    print("All detections are red:", all(detection.metadata.get('colour') == 'red' for detection in all_dets))
    print("Correct number of detections:", len(all_dets) == 6)
    assert(len(all_dets) == 6)


def tes_4():
    rule = lambda detection, min_time, max_time: (min_time <= detection.timestamp + datetime.timedelta(seconds=detection.metadata.get("score", 0.0)) < max_time)
    feeder = ConditionalTimeSteppedFeeder(reader=tes_detections(),
                                          condition=rule,
                                          time_step=datetime.timedelta(seconds=0.1))

    all_dets = []
    for time1, dets1 in feeder:
        all_dets.extend(dets1)

        time0 = datetime.datetime(2019, 4, 1, 14)
        for det in dets1:
            print("Time:", (det.timestamp-time0).total_seconds(), "Score:", det.metadata.get("score", 0.0))
        if len(dets1)>0:
            print("-----------------------------------------")


def tes_5():

    def rule_on_time(detection, min_time, max_time):
        return min_time <= detection.timestamp < max_time

    def is_red(detection, min_time, max_time):
        return detection.metadata.get('colour') == 'red'

    def is_green(detection, min_time, max_time):
        return detection.metadata.get('colour') == 'green'

    def is_blue(detection, min_time, max_time):
        return detection.metadata.get('colour') == 'blue'

    def no_colour(detection, min_time, max_time):
        return detection.metadata.get('colour') == None

    def create_late_rule(time_delay):
        return lambda detection, min_time, max_time: \
            min_time <= detection.timestamp + datetime.timedelta(seconds=time_delay) < max_time

    def rule(d, t1, t2):
        return (no_colour(d, t1, t2) and rule_on_time(d, t1, t2)) or \
               (is_red(d, t1, t2) and create_late_rule(5.0)(d, t1, t2)) or \
               (is_blue(d, t1, t2) and create_late_rule(2.0)(d, t1, t2)) or \
               (is_green(d, t1, t2) and create_late_rule(1.0)(d, t1, t2))

    feeder = ConditionalTimeSteppedFeeder(reader=tes_detections(),
                                          condition=rule,
                                          time_step=datetime.timedelta(seconds=0.1))

    all_dets = []
    for time1, dets1 in feeder:
        all_dets.extend(dets1)

        time0 = datetime.datetime(2019, 4, 1, 14)
        for det in dets1:
            print("Time:", (det.timestamp - time0).total_seconds(), "Colour:",
                  det.metadata.get("colour", "None"))
        if len(dets1) > 0:
            print("-----------------------------------------")

    five=5


if __name__ == '__main__':
    #tes_1()
    #tes_2()
    #tes_3()
    #tes_4()
    tes_5()