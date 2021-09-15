from abc import abstractmethod
from collections import defaultdict

import numpy as np
from scipy.optimize import brute, basinhopping

from . import BruteForceSensorManager
from ..base import Property


class _OptimizeSensorManager(BruteForceSensorManager):

    @abstractmethod
    def _optimiser(self, optimise_func, all_action_generators):
        raise NotImplementedError

    def choose_actions(self, tracks_list, timestamp, nchoose=1, *args, **kwargs):
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
            return -self.reward_function(config, tracks_list, timestamp)

        best_x = self._optimiser(optimise_func, all_action_generators)

        config = config_from_x(best_x)

        return [config]


class OptimizeBruteSensorManager(_OptimizeSensorManager):
    number_of_grid_points: int = Property(
        default=10,
        doc="Number of grid points to search along axis. See Ns in :func:`scipy.optimize.brute`. "
            "Default is 10.")

    def _optimiser(self, optimise_func, all_action_generators):
        ranges = [
            (gen.min, gen.max)
            for gens in all_action_generators.values()
            for gen in gens]
        result = brute(optimise_func,
                       ranges=ranges,
                       Ns=self.number_of_grid_points)
        return np.atleast_1d(result)


class OptimizeBasinHoppingSensorManager(_OptimizeSensorManager):

    def _optimiser(self, optimise_func, all_action_generators):
        initial_values = [
            float(gen.initial_value)
            for gens in all_action_generators.values()
            for gen in gens]
        result = basinhopping(func=optimise_func,
                              x0=initial_values,
                              niter=50,
                              stepsize=1)
        return np.atleast_1d(result.x)
