from collections.abc import Sized, Iterable, Container

import numpy as np

from ..base import Property
from ..functions import gm_reduce_single
from .base import Type
from .array import StateVectors
from .state import TaggedWeightedGaussianState, WeightedGaussianState, GaussianState


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

    @property
    def state_vectors(self):
        if np.any(self.components):
            return StateVectors([component.state_vector for component in self.components])
        else:
            return None

    @property
    def covars(self):
        if np.any(self.components):
            return np.stack([component.covar for component in self.components], axis=2)
        else:
            return None

    @property
    def weights(self):
        if np.any(self.components):
            return np.array([component.weight for component in self.components])
        else:
            return None

    def reduce(self):
        if np.any(self.components):
            state_vectors = []
            covars = []
            weights = []
            for component in self.components:
                state_vectors.append(component.state_vector)
                covars.append(component.covar)
                weights.append(component.weight)
            mean, covar = gm_reduce_single(
                StateVectors(state_vectors), np.stack(covars, axis=2), np.array(weights))
            return GaussianState(mean, covar)
        else:
            return None
