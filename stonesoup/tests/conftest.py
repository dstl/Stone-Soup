# -*- coding: utf-8 -*-
import pytest

from ..base import Base, Property


@pytest.fixture(scope='session')
def base():
    class _TestBase(Base):
        property_a = Property(int)
        property_b = Property(str)
        property_c = Property(int, default=123)
    return _TestBase
