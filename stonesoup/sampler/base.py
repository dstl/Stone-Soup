from ..base import Base


class Sampler(Base):
    """Sampler base class

    A sampler is used to generate samples from a probability distribution specified by the
    user. This class is provided to allow any set of samples to be generated from any specified
    distribution. A :class:`Sampler` should return sub-types of :class:`~.State`.
    """
