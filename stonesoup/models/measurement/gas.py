from typing import Sequence, Union

from math import sqrt
import numpy as np
from scipy.special import erf

from ...base import Property
from ...types.numeric import Probability

from ...types.array import StateVector, CovarianceMatrix, StateVectors
from ..base import GaussianModel
from .base import MeasurementModel
from ...types.state import State


class IsotropicPlume(GaussianModel, MeasurementModel):
    r"""This is a class implementing the isotropic plume model for
    approximating the resulting plume from gas release. Mathematical
    formulation of the algorithm can be seen in [1]_ and [2]_.
    The model assumes isotropic diffusivity and mean wind velocity,
    source strength and turbulent conditions.

    The model calculates the concentration level at a given location
    based on the provided source term. The model employs a sensing
    threshold for deciding if gas has been detected or if the reading
    is just sensor noise and turbulent conditions are accounted for
    using a missed detection probability. The source term if formed
    according to the following:

    .. math::
        \mathbf{S} = \left[\begin{array}{c}
                x \\
                y \\
                z \\
                Q \\
                u \\
                \phi \\
                \zeta_1 \\
                \zeta_2
            \end{array}\right],

    where :math:`x, y` and :math:`z` are the source position in 3D Cartesian
    space, :math:`Q` is the emission rate/strength in g/s, :math:`u` is the
    wind speed in m/s, :math:`\phi` is the wind direction in radians,
    :math:`\zeta_1` is the diffusivity of the gas in the environment and
    :math:`\zeta_2` is the lifetime of the gas.

    The concentration is calculated according to

    .. math::
        \begin{multline}
        \mathcal{M}(\vec{x}_k, \Theta_k) = \frac{Q}{4\pi\Vert\vec{x}_k-\vec{p}^s\Vert}
        \cdot\text{exp}\left[\frac{-\Vert\vec{x}_k-\vec{p}^s\Vert}{\lambda}\right]\cdot\\
        \text{exp}\left[\frac{-(x_k-x)u\cos\phi}{2\zeta_1}\right]
        \cdot\text{exp}\left[\frac{-(y_k-y)u\sin\phi}{2\zeta_1}\right],
        \end{multline}

    where :math:`\vec{x}_k` is the position of the sensor in 3D Cartesian space
    (:math:`[x_k\quad y_k\quad z_k]^\intercal`), :math:`\vec{p}^s` is the source location
    (:math:`[x\quad y\quad z]^\intercal`) and

    .. math::
        \lambda = \sqrt{\frac{\zeta_1\zeta_2}{1+\frac{(u^2\zeta_2)}{4\zeta_1}}}.

    References
    ----------
    .. [1] Vergassola, Massima & Villermaux, Emmanuel & Shraiman, Boris I. "'Infotaxis'
           as a strategy for searching without gradients", Nature, vol. 445, 406-409, 2007
    .. [2] Hutchinson, Michael & Liu, Cunjia & Chen, Wen-Hua, "Source term estimation of
           a hazardous airborne release using an unmanned aerial vehicle", Journal of Field
           Robotics, Vol. 36, 797-917, 2019
    """

    ndim_state: int = Property(
        default=8,
        doc="Number of state dimensions"
    )

    mapping: Sequence[int] = Property(
        default=tuple(range(0, 8)),
        doc="Mapping between measurement and state dims"
    )

    min_noise: float = Property(
        default=1e-4,
        doc="Minimum sensor noise"
    )

    standard_deviation_percentage: float = Property(
        default=0.5,
        doc="Standard deviation as a percentage of the concentration level"
    )

    translation_offset: StateVector = Property(
        default=None,
        doc="A 3x1 array specifying the Cartesian origin offset in terms of :math:`x,y,z` "
            "coordinates.")

    missed_detection_probability: Probability = Property(
        default=0.1,
        doc="The probability that the detection has detection has been affected by turbulence."
    )

    sensing_threshold: float = Property(
        default=0.1,
        doc="Measurement threshold. Should be set high enough to minimise false detections."
    )

    def __init__(self, *args, **kwargs):
        """
        Ensure that the translation offset is initiated
        """
        super().__init__(*args, **kwargs)
        # Set values to defaults if not provided
        if self.translation_offset is None:
            self.translation_offset = StateVector([0] * 3)

    def covar(self, **kwargs) -> CovarianceMatrix:
        raise NotImplementedError('Covariance for IsotropicPlume is dependant on the '
                                  'measurement as well as standard deviation!')

    @property
    def ndim_meas(self) -> int:
        return 1

    def function(self, state: State, noise: Union[bool, np.ndarray] = False, **kwargs) -> Union[
                 StateVector, StateVectors]:
        r"""Model function :math:`h(\vec{x}_t,\vec{v}_t)`

        Parameters
        ----------
        state: :class:`~.StateVector`
            An input source term state vector

        noise: :class:`numpy.ndarray` or bool
            An externally generated random process noise sample (the default is
            `False`, in which case no noise will be added
            if 'True', the output of :meth:`~.Model.rvs` is added). If `False`,
            then the model also does not consider the :attr:`sensing_threshold`
            and :attr:`missed_detection_probability`

        Returns
        -------
        : :class:`numpy.ndarray` of shape (1, 1)
            The model function evaluated with the provided source term
        """

        x, y, z, Q, u, phi, ci, cii = state.state_vector[self.mapping, :].view(np.ndarray)

        px, py, pz = self.translation_offset
        lambda_ = np.sqrt((ci * cii)/(1 + (u**2 * cii)/(4 * ci)))
        abs_dist = np.linalg.norm(state.state_vector[self.mapping[:3], :]
                                  - self.translation_offset, axis=0)

        # prevent divide by zero when converging on the source location
        abs_dist[abs_dist < 0.1] = 0.1

        C = Q / (4 * np.pi * ci * abs_dist) * np.exp(
            (-(px - x) * u * np.cos(phi) / (2 * ci)) + (-(py - y) * u * np.sin(phi) / (2 * ci))
            + (-1 * abs_dist / lambda_))

        C = np.atleast_2d(C)

        if noise:
            C += self.rvs(state=C.view(StateVectors),
                          num_samples=state.state_vector.shape[1],
                          **kwargs)
            # measurement thresholding
            C[C < self.sensing_threshold] = 0
            # missed detections
            flag = np.random.uniform(size=state.state_vector.shape[1]) \
                > (1 - self.missed_detection_probability)
            C[:, flag] = 0

        return C.view(StateVectors)

    def logpdf(self, state1: State, state2: State, **kwargs) -> Union[float, np.ndarray]:
        r"""Model log pdf/likelihood evaluation function

        Evaluates the log pdf/likelihood of ``state1``, given the state
        ``state2`` which is passed to :meth:`function()`.

        This function implements the likelihood functions from
        :meth:`~.pdf` that have been converted to the log space.

        Parameters
        ----------
        state1 : :class:`~.State`
        state2 : :class:`~.State`

        Returns
        -------
        :  float or :class:`~.numpy.ndarray`
            The log likelihood of ``state1``, given ``state2``
        """

        p_m = self.missed_detection_probability
        nd_sigma = self.sensing_threshold
        pred_meas = self.function(state2, **kwargs)
        if state1.state_vector[0] <= self.sensing_threshold:
            pdf = p_m + ((1-p_m) * 1/2 * (1+erf((self.sensing_threshold - pred_meas)
                                                / (nd_sigma * sqrt(2)))))
            likelihood = np.atleast_1d(np.log(pdf)).view(np.ndarray)

        else:
            d_sigma = self.standard_deviation_percentage * pred_meas + self.min_noise
            with np.errstate(divide="ignore"):
                likelihood = np.atleast_1d(np.log(1/(d_sigma*np.sqrt(2*np.pi)) *
                                                  np.exp((-(state1.state_vector-pred_meas)
                                                         ** 2)/(2*d_sigma**2)))).view(np.ndarray)

        if len(likelihood) == 1:
            likelihood = likelihood[0]

        return likelihood

    def pdf(self, state1: State, state2: State, **kwargs) -> Union[Probability, np.ndarray]:
        r"""Model pdf/likelihood evaluation function

        Evaluates the pdf/likelihood of ``state1``, given the state
        ``state2`` which is passed to :meth:`function()`.

        This function implements the following likelihood function,
        adapted from (12) in [2]_, removing the background sensor noise
        term.

        .. math::
            p(z_k|\Theta_k) = \frac{1}{\sigma_d\sqrt{2\pi}}\text{exp}
            \left(-\frac{(z_k-\hat{z}_k)^2}{2\sigma_d}\right),

        where :math:`z_k` = ``state1``, :math:`\Theta_k` = ``state2``,
        :math:`\hat{z}_k` = :meth:`~.Model.function` on ``state2``
        and :math:`\sigma_d` is the measurement standard deviation
        assuming the measurement arose from a true gas detection.
        This is given by

        .. math::
            \sigma_d = \sigma_{\text{percentage}} \cdot \hat{z} + \nu_{\text{min}},

        where :math:`\sigma_{\text{percentage}}` = :attr:`standard_deviation_percentage`
        and :math:`\nu_{\text{min}}` = :attr:`noise`. In the
        event that a measurement is below the sensor threshold or
        missed, a different likelihood function is used. This is
        given by

        .. math::
            p(z_k|\Theta_k) = (P_m) + \left((1-P_m)\cdot\frac{1}{2}\left[1+\text{erf}
            \left(\frac{z_{\text{thr}}-\hat{z}_k}{\sigma_m\sqrt{2}}\right)\right]\right),

        where :math:`P_m` = :attr:`missed_detection_probability`,
        :math:`\sigma_m` is the missed detection standard deviation
        which is implemented as equal to :attr:`sensing_threshold`
        and :math:`\text{erf}()` is the error function.

        Parameters
        ----------
        state1 : :class:`~.State`
        state2 : :class:`~.State`

        Returns
        -------
        : :class:`~.Probability`
            The likelihood of ``state1``, given ``state2``
        """
        return super().pdf(state1, state2, **kwargs)

    def rvs(self, state: Union[StateVector, StateVectors], num_samples: int = 1,
            random_state=None, **kwargs) -> Union[StateVector, StateVectors]:
        r"""Model noise/sample generation function

        Generates noise samples from the model. For this noise, the magnitude
        of sensor noise depends on the measurement. Thus, the noise term is given by

        .. math::
             \nu_k = \mathcal{N}\left(0,(\sigma_{\text{percentage}}\cdot z_k)^2\right).

        Parameters
        ----------
        state: :class:`~.StateVector` or :class:`~.StateVectors`
            The measured state (concentration for this model) used
            to scale the noise term.
        num_samples: scalar, optional
            The number of samples to be generated (the default is 1).

        Returns
        -------
        noise : 2-D array of shape (:attr:`ndim`, ``num_samples``)
            A set of Np samples, generated from the model's noise
            distribution.
        """

        random_state = random_state if random_state is not None else self.random_state

        generator = np.random.RandomState(random_state)
        noise = generator.normal(np.zeros(self.ndim_meas),
                                 np.ravel(state*self.standard_deviation_percentage),
                                 num_samples)

        noise = np.atleast_2d(noise)

        if num_samples == 1:
            return noise.view(StateVector)
        else:
            return noise.view(StateVectors)
