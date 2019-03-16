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
                Detection([[0], [0]], timestamp=time,
                          metadata={'colour': 'red'}),
                Detection([[0], [0]], timestamp=time,
                          metadata={'colour': 'green'}),
                Detection([[0], [0]], timestamp=time,
                          metadata={'colour': 'blue'}),
            }

            time += time_step
            yield time, {
                Detection([[0], [0]], timestamp=time,
                          metadata={'colour': 'red'}),
                Detection([[0], [0]], timestamp=time,
                          metadata={'colour': 'green'}),
                Detection([[0], [0]], timestamp=time,
                          metadata={'colour': 'green'}),
                Detection([[0], [0]], timestamp=time,
                          metadata={'colour': 'blue'}),
            }

            time += time_step
            yield time, {
                Detection([[0], [0]], timestamp=time,
                          metadata={'colour': 'red'}),
                Detection([[0], [0]], timestamp=time - time_step/2,
                          metadata={'colour': 'red'}),
                Detection([[0], [0]], timestamp=time,
                          metadata={'colour': 'blue'}),
            }

            time += time_step
            yield time, {
                Detection([[0], [0]], timestamp=time,
                          metadata={'colour': 'red'}),
                Detection([[0], [0]], timestamp=time,
                          metadata={'colour': 'blue'}),
            }

            time += time_step
            yield time, {
                Detection([[0], [0]], timestamp=time,
                          metadata={'colour': 'red'}),
                Detection([[0], [0]], timestamp=time,
                          metadata={'colour': 'blue'}),
                Detection([[0], [0]], timestamp=time,
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

            time -= time_step*1
            yield time, {
                Detection([[0], [0]], timestamp=time,
                          metadata={'colour': 'red'}),
                Detection([[0], [0]], timestamp=time,
                          metadata={'colour': 'blue'}),
            }

    return Detector()
