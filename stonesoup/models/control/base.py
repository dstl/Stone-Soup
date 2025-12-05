from abc import abstractmethod

from ..base import Model


class ControlModel(Model):
    """Control Model base class"""

    @property
    def ndim(self) -> int:
        return self.ndim_ctrl

    @property
    @abstractmethod
    def ndim_ctrl(self) -> int:
        """Number of control input dimensions"""
        pass

    @abstractmethod
    def function(self, control_input, prior, noise=False, **kwargs):
        r"""Control Model function :math:`f_k(u(k),x(k),w(k))`

        Parameters
        ----------
        control_input : :class:`State`, optional
            :math:`\mathbf{u}_k`
        prior : :class:`~.State`
            The prior state, :math:`\mathbf{x}_{k-1}`
        noise: :class:`numpy.ndarray` or bool
            An externally generated random process noise sample (the default is
            `False`, in which case no noise will be added
            if 'True', the output of :meth:`~.Model.rvs` is used)

        Returns
        -------
        : :class:`StateVector` or :class:`StateVectors`
            The StateVector(s) with the model function evaluated.
        """
        raise NotImplementedError()
