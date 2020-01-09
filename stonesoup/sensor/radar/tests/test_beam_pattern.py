# -*- coding: utf-8 -*-
import datetime

import numpy as np

from ..beam_pattern import StationaryBeam, BeamSweep


def test_stationary_beam():
    # should give same result regardless of time
    dwell_centre = [1, 2]
    time = datetime.datetime.now()
    beam_position = StationaryBeam(centre=dwell_centre)
    current_pos = beam_position.move_beam(time)
    assert current_pos == dwell_centre
    time += datetime.timedelta(seconds=0.5)
    new_pos = beam_position.move_beam(time)
    assert new_pos == dwell_centre


def test_beam_sweep():

    start_time = datetime.datetime.now()

    beam_pattern = BeamSweep(angle_per_s=np.pi/18,
                             centre=[np.pi/45, np.pi/45],
                             frame=[np.pi/9, np.pi/18],
                             init_time=start_time,
                             separation=np.pi/36)

    # should start in top left corner
    assert beam_pattern.move_beam(start_time) == [-np.pi/30, np.pi/20]
    # move in azimuth
    assert beam_pattern.move_beam(start_time + datetime.timedelta(seconds=1)) \
        == [np.pi/45, np.pi/20]
    # moved to next elevation
    az, el = beam_pattern.move_beam(start_time + datetime.timedelta(seconds=3))
    assert [round(az, 10), round(el, 10)] == [round(np.pi/45, 10),
                                              round(np.pi/45, 10)]
    # restart frame and moved in azimuth
    az, el = beam_pattern.move_beam(start_time + datetime.timedelta(seconds=7))
    assert [round(az, 10), round(el, 10)] == [round(np.pi/45, 10),
                                              round(np.pi/20, 10)]
