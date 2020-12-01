import numpy as np

from collections.abc import Sized, Iterable, Container
from typing import MutableSequence

from .base import Type
from ..base import Property
from .state import TaggedWeightedGaussianState, WeightedGaussianState
from .array import StateVectors
from .numeric import Probability
from ..functions import gm_reduce_single


class GaussianMixture(Type, Sized, Iterable, Container):
    """
    Gaussian Mixture type

    Represents the target space through a Gaussian Mixture. Individual Gaussian
    components are contained in a :class:`list` of
    :class:`WeightedGaussianState`.
    """

    components: MutableSequence[WeightedGaussianState] = Property(
        default=None,
        doc="""The initial list of :class:`WeightedGaussianState` components.
        Default `None` which initialises with empty list.""")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.components is None:
            self.components = []
        if any(not isinstance(component, (WeightedGaussianState, TaggedWeightedGaussianState))
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
        return iter(self.components)

    def __getitem__(self, index):
        # retrieve WeightedGaussianState by array index
        return self.components[index]

    def __setitem__(self, index, value):
        return self.components.__setitem__(index, value)

    def __len__(self):
        return len(self.components)

    def append(self, component):
        return self.components.append(component)

    def extend(self, new_components):
        return self.components.extend(new_components)

    @property
    def ndim(self):
        if len(self.components):
            return self.components[0].ndim
        return 0

    @property
    def means(self):
        return StateVectors([component.mean for component in self.components])

    @property
    def covars(self):
        return np.stack([component.covar for component in self.components], axis=2)

    @property
    def weights(self):
        return np.asarray([component.weight for component in self.components])

    @property
    def state_vector(self):
        return self.mean

    @property
    def mean(self):
        means = self.means
        weights = self.weights / Probability.sum(self.weights)
        return np.average(means, axis=1, weights=weights)

    @property
    def covar(self):
        _, covar = gm_reduce_single(self.means, self.covars,
                                    self.weights)
        return covar

    @property
    def component_tags(self):
        component_tags = set()
        if all(isinstance(component, TaggedWeightedGaussianState)
               for component in self.components):
            for component in self.components:
                component_tags.add(component.tag)
        else:
            raise ValueError("All components must be "
                             "TaggedWeightedGaussianState!")
        return component_tags
