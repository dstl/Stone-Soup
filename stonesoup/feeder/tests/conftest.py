# -*- coding: utf-8 -*-
import datetime

import pytest

from ...types.detection import Detection


@pytest.fixture()
def detector():
    class Detector:
        def detections_gen(self):
            time = datetime.datetime(2019, 4, 1, 14)
            time_step = datetime.timedelta(seconds=1)

            yield time, {
                Detection([[50], [0]], timestamp=time,
                          metadata={'colour': 'red'}),
                Detection([[20], [5]], timestamp=time,
                          metadata={'colour': 'green'}),
                Detection([[1], [1]], timestamp=time,
                          metadata={'colour': 'blue'}),
            }

            time += time_step
            yield time, {
                Detection([[-5], [4]], timestamp=time,
                          metadata={'colour': 'red'}),
                Detection([[11], [200]], timestamp=time,
                          metadata={'colour': 'green'}),
                Detection([[0], [0]], timestamp=time,
                          metadata={'colour': 'green'}),
                Detection([[-43], [-10]], timestamp=time,
                          metadata={'colour': 'blue'}),
            }

            time += time_step
            yield time, {
                Detection([[561], [10]], timestamp=time,
                          metadata={'colour': 'red'}),
                Detection([[1], [-10]], timestamp=time - time_step/2,
                          metadata={'colour': 'red'}),
                Detection([[-11], [-50]], timestamp=time,
                          metadata={'colour': 'blue'}),
            }

            time += time_step
            yield time, {
                Detection([[1], [-5]], timestamp=time,
                          metadata={'colour': 'red'}),
                Detection([[1], [-5]], timestamp=time,
                          metadata={'colour': 'blue'}),
            }

            time += time_step
            yield time, {
                Detection([[-11], [5]], timestamp=time,
                          metadata={'colour': 'red'}),
                Detection([[13], [654]], timestamp=time,
                          metadata={'colour': 'blue'}),
                Detection([[-3], [6]], timestamp=time,
                          metadata={}),
            }

            time += time_step*2
            yield time, {
                Detection([[0], [0]], timestamp=time,
                          metadata={'colour': 'red'}),
                Detection([[0], [0]], timestamp=time,
                          metadata={'colour': 'blue'}),
                Detection([[0], [0]], timestamp=time,
                          metadata={}),
                Detection([[0], [0]], timestamp=time,
                          metadata={}),
            }

            time -= time_step
            yield time, {
                Detection([[5], [-6]], timestamp=time,
                          metadata={'colour': 'red'}),
                Detection([[10], [0]], timestamp=time,
                          metadata={'colour': 'blue'}),
            }

    return Detector()
