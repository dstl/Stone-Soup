import pytest

import numpy as np

from ..shape import Area, AreaOfInterest


def test_area_valid_bounds():
    area = Area(xmin=0.0, xmax=5.0, ymin=-1.0, ymax=1.0)

    assert area.xmin == 0.0
    assert area.xmax == 5.0
    assert area.ymin == -1.0
    assert area.ymax == 1.0
    assert isinstance(area, Area)


def test_area_invalid_x_bounds_raises_value_error():
    with pytest.raises(ValueError, match="Area must have xmin < xmax and ymin < ymax"):
        Area(xmin=5.0, xmax=5.0, ymin=0.0, ymax=10.0)


def test_area_invalid_y_bounds_raises_value_error():
    with pytest.raises(ValueError, match="Area must have xmin < xmax and ymin < ymax"):
        Area(xmin=0.0, xmax=10.0, ymin=10.0, ymax=10.0)


def test_area_of_interest_default_properties():
    aoi = AreaOfInterest()

    assert aoi.xmin == -np.inf
    assert aoi.xmax == np.inf
    assert aoi.ymin == -np.inf
    assert aoi.ymax == np.inf
    assert aoi.interest == 1
    assert aoi.access == 1
    assert isinstance(aoi, Area)


def test_area_of_interest_valid_properties():
    aoi = AreaOfInterest(xmin=-5.0, xmax=5.0, ymin=-5.0, ymax=5.0,
                         interest=7, access=3)

    assert aoi.interest == 7
    assert aoi.access == 3
    assert aoi.xmin == -5.0
    assert aoi.xmax == 5.0
    assert aoi.ymin == -5.0
    assert aoi.ymax == 5.0


def test_area_of_interest_invalid_bounds_raises_value_error():
    with pytest.raises(ValueError, match="Area must have xmin < xmax and ymin < ymax"):
        AreaOfInterest(xmin=5.0, xmax=0.0, ymin=0.0, ymax=10.0)
