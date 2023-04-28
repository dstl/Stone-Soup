from ..base import Base


class Sampler(Base):
    """Sampler base class

    A sampler is used to generate discrete samples from a continuous distribution specified by the
    user. This class is intends to generate any form of sample from any kind from any specified
    distribution, depending on what is required. A sample may culminate in different forms but the
    most common would be a :class:`~.State` type.
    """
