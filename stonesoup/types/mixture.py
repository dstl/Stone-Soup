from collections.abc import Sized, Iterable, Container

from .base import Type
from ..base import Property
from .state import WeightedGaussianState


class GaussianMixture(Type, Sized, Iterable, Container):
    """
    Gaussian Mixture type

    Represents the target space through a Gaussian Mixture. Individual Gaussian
    components are contained in a :class:`list` of
    :class:`WeightedGaussianState`.
    """

    components = Property(
        [WeightedGaussianState],
        default=None,
        doc="""The initial list of :class:`WeightedGaussianState` components.
        Default `None` which initialises with empty list.""")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.components is None:
            self.components = []
        if len(self.components) > 0:
            if any(not isinstance(component, WeightedGaussianState)
                    for component in self.components):
                raise ValueError("Cannot form GaussianMixtureState out of "
                                 "non-WeightedGaussianState inputs!")

    def __contains__(self, index):
        # check if 'components' contains any WeightedGaussianState
        # matching 'index'
        if isinstance(index, WeightedGaussianState):
            return index in self.components
        else:
            raise ValueError("Index must be WeightedGaussianState")

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < len(self.components):
            component = self.components[self.index]
            self.index += 1
            return component
        else:
            raise StopIteration

    def __getitem__(self, index):
        # retrieve WeightedGaussianState by array index
        if isinstance(index, int):
            return self.components[index]
        else:
            raise ValueError("Index must be int")

    def __setitem__(self, index, value):
        return self.components.__setitem__(index, value)

    def __len__(self):
        return len(self.components)

    def append(self, component):
        return self.components.append(component)
