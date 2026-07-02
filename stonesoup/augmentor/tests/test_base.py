import pytest

from ..base import Augmentor


def test_baseaugmentor():
    with pytest.raises(TypeError):
        Augmentor()
