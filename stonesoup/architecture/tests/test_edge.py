import pytest

from datetime import datetime

from ..edge import Edges, DataPiece
from ...types.track import Track


def test_data_piece(nodes, times):
    with pytest.raises(TypeError):
        data_piece_fail = DataPiece()

    data_piece = DataPiece(node=nodes['a'], originator=nodes['a'],
                           data=Track([]), time_arrived=times['a'])
    assert data_piece.sent_to == set()

def test_bleh():
    print(datetime.now().strftime(format="%d/%m/%Y %H:%M:%S"))
