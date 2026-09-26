import numpy as np
from ..state import Mahalanobis, SquaredMahalanobis


def test_mahalanobis_clamps_negative_roundoff(monkeypatch):
    monkeypatch.setattr(
        SquaredMahalanobis,
        "__call__",
        lambda self, state1, state2: np.array([-1e-12, 4.0]),
    )

    result = Mahalanobis()(None, None)

    assert np.allclose(result, [0.0, 2.0])


def test_mahalanobis_clamps_negative_distance(monkeypatch):
    monkeypatch.setattr(
        SquaredMahalanobis,
        "__call__",
        lambda self, state1, state2: -1e-6,
    )

    assert Mahalanobis()(None, None) == 0.0
