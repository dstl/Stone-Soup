# -*- coding: utf-8 -*
import pytest
import numpy as np
from PIL import Image

from stonesoup.reader.image import SingleImageFileReader


@pytest.fixture()
def img_gt_filename(tmpdir):
    img_filename = tmpdir.join("test.png")
    imarray = np.random.rand(100, 100, 3) * 255
    im = Image.fromarray(imarray.astype('uint8')).convert('RGB')
    im.save(img_filename.strpath)
    return img_filename


def test_single_image_file_reader(img_gt_filename):
    reader = SingleImageFileReader(img_gt_filename.strpath)
    # timestamp, frame = next(reader)
    for timestamp, frame in reader:
        im = Image.open(img_gt_filename.strpath)
        img = np.array(im)
        assert timestamp is None
        assert np.array_equal(frame.pixels, img)
