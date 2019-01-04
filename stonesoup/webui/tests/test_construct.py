# -*- coding: utf-8 -*-

from ..construct import construct


def test_numpy():
    comap = construct({
        '__class__': "stonesoup.types.state.State",
        'timestamp': "2018-01-01T12:01:00",
        'state_vector': [[0], [1], [2]],
    })

    assert comap
