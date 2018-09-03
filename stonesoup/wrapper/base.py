import stonesoup
from ..base import Base, Property


class Wrapper(Base):
    """Wrapper base class

    Wrappers are used to run code not written in python and convert
    the outputs back into Stone Soup objects.
    """

    directory_path = Property(str,
                              doc='Top level location of module. Defaults'
                                  ' to Stone Soup install location',
                              default=stonesoup.__file__.strip('__init__.py'))
