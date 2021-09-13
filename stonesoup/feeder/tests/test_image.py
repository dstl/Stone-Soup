import pickle
import numpy as np

from stonesoup.reader.image import SingleImageFileReader
from stonesoup.feeder.image import CFAR, CCL


def test_cfar_detections(datadir):
    input_filename = datadir.join('test_img.png')
    reader = SingleImageFileReader(input_filename)
    feeder = CFAR(reader, train_size=10, guard_size=4, alpha=4, squared=True)
    for _, frame in feeder:
        th1 = feeder.current[1].pixels
    expected_result_filename = datadir.join('expected_result_cfar.pickle')
    file = open(expected_result_filename, 'rb')
    img = pickle.load(file)
    assert np.array_equal(th1, img)


def test_ccl_detections(datadir):
    input_filename = datadir.join('test_img.png')
    reader = SingleImageFileReader(input_filename)
    cfar = CFAR(reader, train_size=10, guard_size=4, alpha=4, squared=True)
    feeder = CCL(cfar)
    for _, frame in feeder:
        labels_img = frame.pixels
    expected_result_filename = datadir.join('expected_result_ccl.pickle')
    file = open(expected_result_filename, 'rb')
    img2 = pickle.load(file)
    assert np.array_equal(labels_img, img2)
