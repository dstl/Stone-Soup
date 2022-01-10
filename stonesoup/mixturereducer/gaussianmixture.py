# -*- coding: utf-8 -*-
from scipy.spatial import distance as dist
import uuid

from ..base import Property
from .base import MixtureReducer
from ..types.state import TaggedWeightedGaussianState, WeightedGaussianState
from operator import attrgetter


class GaussianMixtureReducer(MixtureReducer):
    """
    Gaussian Mixture Reducer class:

    Reduces the number of components in a Gaussian mixture to increase
    computational efficiency. See [1] for details.
    Achieved in two ways: pruning and merging.
    Pruning is the act of removing low weight components from the mixture
    that fall below a pruning threshold.
    Merging is the act of combining similar components in the mixture
    that fall with a distance threshold into a single component.

    References
    ----------
    [1] B.-N. Vo and W.-K. Ma, “The Gaussian Mixture Probability Hypothesis
    Density Filter,” Signal Processing,IEEE Transactions on, vol. 54, no. 11,
    pp. 4091–4104, 2006..
    """

    prune_threshold: float = Property(default=1e-9, doc="Threshold for pruning.")
    merge_threshold: float = Property(default=16, doc='Threshold for merging')
    merging: bool = Property(default=True, doc='Flag for merging')
    pruning: bool = Property(default=True, doc='Flag for pruning')

    def reduce(self, components_list):
        """
        Reduce the components of Gaussian Mixture :class:`list`
        through pruning and merging

        Parameters
        ----------
        components_list : :class:`~.list`
            The components of Gaussian Mixture

        Returns
        -------
        :class:`~.list`
            Reduced components

            """
        if len(components_list) > 0:
            if self.pruning:
                components_list = self.prune(components_list)
            if len(components_list) > 1 & self.merging:
                components_list = self.merge(components_list)
        return components_list

    def prune(self, components_list):
        """
        Pruning is the act of removing low weight components from the mixture
        that fall below a pruning threshold :attr:`prune_threshold`.

        Parameters
        ----------
        components_list : :class:`~.list`
            The components of Gaussian Mixture to be pruned

        Returns
        -------
        remaining_components : :class:`~.GaussianMixtureState`
             Components that remain after pruning

        """
        # Prune low weight components
        pruned_weight_sum = 0
        for component in components_list:
            if component.weight < self.prune_threshold:
                pruned_weight_sum += component.weight

        remaining_components = [component for component in components_list
                                if component.weight > self.prune_threshold]
        # Distribute pruned weights across remaining components
        for component in remaining_components:
            component.weight += \
                pruned_weight_sum / len(remaining_components)
        return remaining_components

    def merge_components(self, component_1, component_2):
        """
        Merge two similar components

        Parameters
        ----------
        component_1 : :class:`~.WeightedGaussianState`
            First component to be merged
        component_2 : :class:`~.WeightedGaussianState`
            Second component to be merged

        Returns
        -------
        merged_component : :class:`~.WeightedGaussianState`
            Merged Gaussian component

        """
        weight_sum = component_1.weight+component_2.weight
        w1 = component_1.weight / weight_sum
        w2 = component_2.weight / weight_sum
        merged_mean = component_1.mean*w1 + component_2.mean*w2
        merged_covar = component_1.covar*w1 + component_2.covar*w2
        mu1_minus_m2 = component_1.mean - component_2.mean
        merged_covar = merged_covar + \
            mu1_minus_m2*mu1_minus_m2.T*w1*w2
        merged_weight = component_1.weight + component_2.weight
        if merged_weight > 1:
            merged_weight = 1
        if isinstance(component_1, TaggedWeightedGaussianState):
            merged_component = TaggedWeightedGaussianState(
                state_vector=merged_mean,
                covar=merged_covar,
                weight=merged_weight,
                tag=component_1.tag,
                timestamp=component_1.timestamp
            )
        elif isinstance(component_1, WeightedGaussianState):
            merged_component = WeightedGaussianState(
                state_vector=merged_mean,
                covar=merged_covar,
                weight=merged_weight,
                timestamp=component_1.timestamp
            )

        return merged_component

    def merge(self, components_list):
        """
        Merging is the act of combining similar components in the mixture
        that fall with a distance threshold :attr:`merge_threshold` into
        a single component.

        Parameters
        ----------
        components_list : :class:`~.list`
            Components of the Gaussian Mixture to be merged

        Returns
        -------
        :class:`~.list`
            Merged components

        """
        # Sort components by weight
        remaining_components = sorted(
            components_list, key=attrgetter('weight'))

        merged_components = []
        final_merged_components = []
        while remaining_components:
            # Get highest weighted component
            best_component = remaining_components.pop()
            # Check for similar components
            # (modifying list in loop, so copy used)
            for component in remaining_components.copy():
                # Calculate distance between component and best component
                distance = dist.mahalanobis(
                    best_component.mean[:, 0], component.mean[:, 0], best_component.covar)
                # Merge if similar
                if distance < self.merge_threshold:
                    remaining_components.remove(component)
                    best_component = self.merge_components(
                        best_component, component
                    )
            # Add potentially merged component to new mixture
            merged_components.append(best_component)
        if all(isinstance(component, TaggedWeightedGaussianState)
               for component in merged_components):
            # Check for duplicate tags
            components_tags = set(component.tag for component in merged_components)
            if len(components_tags) != len(merged_components):
                # There are duplicatze tags so assign
                # new tags to the lower weighted shared ones
                for shared_tag in components_tags:
                    shared_components = sorted(
                        (component for component in merged_components
                            if component.tag == shared_tag),
                        key=attrgetter('weight'),
                        reverse=True)
                    final_merged_components.append(shared_components[0])
                    for component in shared_components[1:]:
                        # Assign a new uuid
                        component.tag = str(uuid.uuid4())
                        final_merged_components.append(component)
            else:
                # No duplicates
                final_merged_components.extend(merged_components)
        else:
            # Just weighted components (no tags)
            final_merged_components.extend(merged_components)
        # Assign merged components to the mixture
        return final_merged_components
