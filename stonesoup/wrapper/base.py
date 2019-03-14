import stonesoup
from ..base import Base, Property


class Wrapper(Base):
    """Wrapper base class

    Wrappers are used to run code not written in python and convert
    the outputs back into Stone Soup objects.
    """

    dir_path = Property(
        str, default=None,
        doc='Top level location of module. Defaults to Stone Soup install location')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if(self.dir_path is None):
            self.dir_path = stonesoup.__file__.strip('__init__.py')
