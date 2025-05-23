from abc import abstractmethod
from stonesoup.base import Base


class Proposal(Base):

    @abstractmethod
    def rvs(self, *args, **kwargs):
        r"""Proposal noise/sample generation function

        Generates samples from the proposal.

        Parameters
        ----------
        state: :class:`~.State`
            The state to generate samples from.
        """
        raise NotImplementedError
