# -*- coding: utf-8 -*-
import pytest

from ..base import Base, Property


@pytest.fixture(scope='session')
def base():
    class _TestBase(Base):
        property_a: int = Property()
        property_b: str = Property()
        property_c: int = Property(default=123)
    return _TestBase
