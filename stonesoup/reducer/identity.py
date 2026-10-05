from .base import Reducer


class IdentityReducer(Reducer):
    """Reducer that preserves the current state mixture without reduction.

    The IdentityReducer applies only likelihood recalculation to the input
    state mixture and returns it unchanged otherwise. This reducer is useful in
    situations where no further hypothesis pruning or model merging is required.
    """

    def reduce(self, states, timestamp):
        """Return states after recalculating likelihoods.

        Parameters
        ----------
        states : :class:`~.GaussianMixture` of :class:`~.ModelAugmentedWeightedGaussianState`
            Input state mixture to evaluate.
        timestamp : datetime.datetime
            Timestamp used to compute likelihoods for the mixture.

        Returns
        -------
        :class:`~.GaussianMixture` of :class:`~.ModelAugmentedWeightedGaussianState`
            The input states with updated likelihood weights.
        """
        states = self.calculate_likelihood(states, timestamp)
        return states
