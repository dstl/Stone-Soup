# -*- coding: utf-8 -*-
import os
import datetime
from distutils import dir_util
import pytest

from ...buffered_generator import BufferedGenerator
from ...reader import DetectionReader, GroundTruthReader
from ...types.detection import Detection
from ...types.groundtruth import GroundTruthPath, GroundTruthState


@pytest.fixture(params=['detector', 'groundtruth'])
def reader(request):
    return request.getfixturevalue(request.param)


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


@pytest.fixture()
def groundtruth():
    class GroundTruth(GroundTruthReader):
        @BufferedGenerator.generator_method
        def groundtruth_paths_gen(self):

            time = datetime.datetime(2020, 1, 1, 0)
            time_step = datetime.timedelta(seconds=1)
            state = GroundTruthState(state_vector=[[0], [0]],
                                     timestamp=time,
                                     metadata={'colour': 'red',
                                               'score': 0})
            redpath = GroundTruthPath(id='red')
            redpath.append(state)
            state = GroundTruthState(state_vector=[[0], [0]],
                                     timestamp=time,
                                     metadata={'colour': 'yellow',
                                               'score': 0})
            yellowpath = GroundTruthPath(id='yellow')
            yellowpath.append(state)
            yield time, {redpath, yellowpath}

            time += time_step
            state = GroundTruthState(state_vector=[[1], [0]],
                                     timestamp=time,
                                     metadata={'colour': 'red',
                                               'score': 2})
            redpath.append(state)
            state = GroundTruthState(state_vector=[[0], [101]],
                                     timestamp=time,
                                     metadata={'colour': 'yellow',
                                               'score': 5})
            yellowpath.append(state)
            yield time, {redpath, yellowpath}

            time -= time_step
            state = GroundTruthState(state_vector=[[101], [0]],
                                     timestamp=time,
                                     metadata={'colour': 'red',
                                               'score': 3})
            redpath.append(state)
            state = GroundTruthState(state_vector=[[0], [101]],
                                     timestamp=time,
                                     metadata={'colour': 'yellow',
                                               'score': 10})
            yellowpath.append(state)
            yield time, {redpath, yellowpath}

    return GroundTruth()


@pytest.fixture
def datadir(tmpdir, request):
    '''
    Fixture responsible for searching a folder with the same name of test
    module and, if available, moving all contents to a temporary directory so
    tests can use them freely.
    '''
    filename = request.module.__file__
    test_dir, _ = os.path.splitext(filename)

    if os.path.isdir(test_dir):
        dir_util.copy_tree(test_dir, str(tmpdir))

    return tmpdir
