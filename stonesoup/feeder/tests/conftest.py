# -*- coding: utf-8 -*-
import datetime

import pytest

from ...buffered_generator import BufferedGenerator
from ...reader import DetectionReader
from ...types.detection import Detection


@pytest.fixture()
def detector():
    class Detector(DetectionReader):
        @BufferedGenerator.generator_method
        def detections_gen(self):
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
                Detection([[1], [-10]], timestamp=time - time_step/2,
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

            time += time_step*2
            yield time, {
                Detection([[0], [0]], timestamp=time,
                          metadata={'colour': 'red',
                                    'score': 1}),
                Detection([[0], [0]], timestamp=time,
                          metadata={'colour': 'blue',
                                    'score': 0.612}),
                Detection([[0], [0]], timestamp=time,
                          metadata={'score': 0}),
                Detection([[0], [0]], timestamp=time,
                          metadata={}),
            }

            time -= time_step
            yield time, {
                Detection([[5], [-6]], timestamp=time,
                          metadata={'colour': 'red',
                                    'score': 0.2}),
                Detection([[10], [0]], timestamp=time,
                          metadata={'colour': 'blue'}),
            }

    return Detector()
