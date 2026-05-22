from ..augmentor.base import Augmentor


class IdentityAugmentor(Augmentor):
    """Identity augmentor that passes states through without modification.

    This augmentor implements the no augmentation strategy. It is useful when
    a filtering pipeline requires an augmentor interface but no model history
    augmentation is desired.
    """

    def augment(self, states):
        """Return input states unchanged.

        Parameters
        ----------
        states : :class:`~.GaussianMixture` of :class:`~.ModelAugmentedWeightedGaussianState`
            Input states that are passed through without changes.

        Returns
        -------
        :class:`~.GaussianMixture` of :class:`~.ModelAugmentedWeightedGaussianState`
            The input states.
        """
        return states
