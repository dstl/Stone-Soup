from abc import abstractmethod
from collections import OrderedDict

from scipy.optimize import brute, basinhopping, fmin

from . import BruteForceSensorManager
from ..base import Property
from ..types.state import StateVector


class _OptimizeSensorManager(BruteForceSensorManager):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def _optimiser(self, optimise_func, all_action_generators, config_from_x):
        raise NotImplementedError

    def choose_actions(self, tracks, timestamp, nchoose=1, return_reward=False, **kwargs):
        if nchoose > 1:
            raise ValueError("Can only return best result (nchoose=1)")

        all_action_generators = OrderedDict([(actionable, actionable.actions(timestamp))
                                            for actionable in self.actionables])

        def config_from_x(x):
            config = OrderedDict()
            i = 0
            for actionable, generators in all_action_generators.items():
                config[actionable] = list()
                for generator in generators:
                    if generator.ndim > 1:
                        ndim_state = generator.ndim
                        action = generator.action_from_value(StateVector(x[i:i+ndim_state]))
                        i += ndim_state

                    else:
                        action = generator.action_from_value(x[i])
                        i += 1

                    config[actionable].append(action)

            return config

        def optimise_func(x):
            config = config_from_x(x)
            if config:
                return -self.reward_function(config, tracks, timestamp)
            else:
                return 0

        best_x = self._optimiser(optimise_func, all_action_generators, config_from_x)
        config = config_from_x(best_x)

        if return_reward:
            reward = self.reward_function(config, tracks, timestamp)
            return [config], reward
        else:
            return [config]


class OptimizeBruteSensorManager(_OptimizeSensorManager):
    # TODO: Rename to Grid based SM?
    """
    A sensor manager built around the SciPy :func:`~.scipy.optimize.brute` method.
    The sensor manager
    takes all possible configurations of sensors and actions and
    uses
    the optimising function to optimise a given reward function,
    returning the optimal configuration.

    `Scipy optimize <https://docs.scipy.org/doc/scipy/reference/optimize.html#utilities>`_ provides
    functions which can minimize or maximize functions using a variety
    of algorithms. The :func:`~.scipy.optimize.brute` minimizes a function over a given range,
    using a brute force method. This is done by computing the function's value at each point of
    a multidimensional grid of points, to find the global minimum.

    A default version of the optimiser is used, or on initiation the sensor manager can be passed
    some parameters to alter the configuration of the optimiser.
    Please see the Scipy documentation site for full details on what each parameter does.
    """

    n_grid_points: int = Property(
        default=10,
        doc="Number of grid points to search along (each) axis. See Ns in "
            ":func:`~.scipy.optimize.brute`. "
            "Default is 10.")
    generate_full_output: bool = Property(default=False,
                                          doc="If True, returns the evaluation grid "
                                              "and the objective "
                                              "function's values on it.")
    finish: bool = Property(default=False,
                            doc="A polishing function can be applied to the result of brute "
                                "force minimisation. If True this is set as "
                                ":func:`~.scipy.optimize.fmin` which "
                                "minimizes a function using the downhill simplex algorithm."
                                "As a default no polishing function is applied.")
    disp: bool = Property(default=False,
                          doc="Set to True to print convergence messages from the finish "
                              "callable.")

    def _optimiser(self, optimise_func, all_action_generators, config_from_x):

        ranges = []

        for gens in all_action_generators.values():
            for generator in gens:
                # if generator dimensions > 1, iterate over until added every min and max
                if generator.ndim > 1:
                    for i in range(generator.ndim):
                        ranges.append((generator.min[i], generator.max[i]))
                else:
                    ranges.append((generator.min, generator.max))

        if self.finish:
            self.finish_func = fmin
        else:
            self.finish_func = None

        result = brute(optimise_func,
                       ranges=ranges,
                       Ns=self.n_grid_points,
                       full_output=self.generate_full_output,
                       finish=self.finish_func,
                       disp=self.disp)

        if self.generate_full_output:
            self.full_output = result
            result = result[0]

        return result

    def get_full_output(self):
        """
        Returns the output generated when `generate_full_output=True` for the most recent
        time step.
        This returns the evaluation grid and reward function's values on it, as generated by the
        :meth:`optimize.brute` method.
        See
        `Scipy documentation
        <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brute.html#scipy.optimize.brute>`_
        for full details.

        Returns
        -------
        full_output: tuple
        """
        if self.full_output:
            return self.full_output


class OptimizeBasinHoppingSensorManager(_OptimizeSensorManager):
    """
    A sensor manager built around the SciPy :meth:`optimize.basinhopping` method.
    The sensor manager
    takes all possible configurations of sensors and actions and
    uses
    the optimising function to optimise a given reward function,
    returning the optimal configuration for the sensing system.

    The :func:`~.scipy.optimize.basinhopping` finds the global minimum of a function using the
    basin-hopping algorithm. This is a combination of a global stepping algorithm and local
    minimization at each step.

    A default version of the optimiser is used, or on initiation the sensor manager can be passed
    some parameters to alter the configuration of the optimiser.
    Please see the Scipy documentation site for full details on what each parameter does.
    """

    n_iter: int = Property(default=100,
                           doc='The number of basin hopping iterations.')
    T: float = Property(default=1.0,
                        doc='The "temperature" parameter for the accept or reject criterion. '
                            'Higher temperatures mean larger jumps in function value will be '
                            'accepted.')
    stepsize: float = Property(default=0.5,
                               doc='Maximum step size for use in the random displacement.')
    interval: int = Property(default=50,
                             doc='Interval for how often to update the stepsize.')
    disp: bool = Property(default=False,
                          doc='Set to True to print status messages.')
    niter_success: int = Property(default=None,
                                  doc='Stop the run if the global minimum candidate '
                                      'remains the same '
                                      'for this number of iterations.')

    def _optimiser(self, optimise_func, all_action_generators, config_from_x):
        initial_values = list()

        for gens in all_action_generators.values():
            for generator in gens:
                # if generator dim > 1, iterate over until added every initial value
                if generator.ndim > 1:
                    for i in range(generator.ndim):
                        initial_values.append(float(generator.initial_value[i]))
                else:
                    initial_values.append(float(generator.initial_value))

        result = basinhopping(func=optimise_func,
                              x0=initial_values,
                              niter=self.n_iter,
                              T=self.T,
                              stepsize=self.stepsize,
                              interval=self.interval,
                              disp=self.disp,
                              niter_success=self.niter_success)
        return result.x
