import pytest

from ..base import Base, Property


class _TestBase(Base):
    property_a: int = Property()
    property_b: str = Property()
    property_c: int = Property(default=123)


@pytest.fixture(scope='session')
def base():
    return _TestBase
