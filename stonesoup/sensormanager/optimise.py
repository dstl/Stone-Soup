from abc import abstractmethod
from collections import defaultdict

import numpy as np
from scipy.optimize import brute, basinhopping, fmin

from . import BruteForceSensorManager
from ..base import Property


class _OptimizeSensorManager(BruteForceSensorManager):

    @abstractmethod
    def _optimiser(self, optimise_func, all_action_generators):
        raise NotImplementedError

    def choose_actions(self, tracks, timestamp, nchoose=1, **kwargs):
        if nchoose > 1:
            raise ValueError("Can only return best result (nchoose=1)")
        all_action_generators = dict()

        for sensor in self.sensors:
            action_generators = sensor.actions(timestamp)
            all_action_generators[sensor] = action_generators

        def config_from_x(x):
            config = defaultdict(list)
            for i, (sensor, generators) in enumerate(all_action_generators.items()):
                for generator in generators:
                    action = generator.action_from_value(x[i])
                    if action is not None:
                        config[sensor].append(action)
            return config

        def optimise_func(x):
            config = config_from_x(x)
            return -self.reward_function(config, tracks, timestamp)

        best_x = self._optimiser(optimise_func, all_action_generators)
        config = config_from_x(best_x)

        return [config]


class OptimizeBruteSensorManager(_OptimizeSensorManager):
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
        doc="Number of grid points to search along axis. See Ns in "
            ":func:`~.scipy.optimize.brute`. "
            "Default is 10.")
    full_output: bool = Property(default=False,
                                 doc="If True, returns the evaluation grid and the objective "
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

    def _optimiser(self, optimise_func, all_action_generators):
        ranges = [
            (gen.min, gen.max)
            for gens in all_action_generators.values()
            for gen in gens]

        if self.finish:
            self.finish_func = fmin
        else:
            self.finish_func = None

        result = brute(optimise_func,
                       ranges=ranges,
                       Ns=self.n_grid_points,
                       full_output=self.full_output,
                       finish=self.finish_func,
                       disp=self.disp)

        if self.full_output:
            print('Full output:', result)
            result = result[0]

        return np.atleast_1d(result)


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

    def _optimiser(self, optimise_func, all_action_generators):
        initial_values = [
            float(gen.initial_value)
            for gens in all_action_generators.values()
            for gen in gens]
        result = basinhopping(func=optimise_func,
                              x0=initial_values,
                              niter=self.n_iter,  # was 50
                              T=self.T,
                              stepsize=self.stepsize,  # was 1
                              interval=self.interval,
                              disp=self.disp,
                              niter_success=self.niter_success)
        return np.atleast_1d(result.x)
