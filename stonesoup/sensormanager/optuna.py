import warnings
from collections import defaultdict
from collections.abc import Iterable

try:
    import optuna
except ImportError as error:
    raise ImportError("Usage of Optuna Sensor Manager requires that the optional package "
                      "`optuna`is installed") from error

from ..base import Property
from ..sensor.sensor import Sensor
from .action import RealNumberActionGenerator, Action
from . import SensorManager


class OptunaSensorManager(SensorManager):
    """Sensor Manager that uses the optuna package to determine the best actions available within
    a time frame specified by :attr:`timeout`."""
    timeout: float = Property(
        doc="Number of seconds that the sensor manager should optimise for each time-step",
        default=10.)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        optuna.logging.set_verbosity(optuna.logging.CRITICAL)

    def choose_actions(self, tracks, timestamp, nchoose=1, **kwargs) -> Iterable[tuple[Sensor,
                                                                                       Action]]:
        """Method to find the best actions for the given :attr:`sensors` to according to the
        :attr:`reward_function`.

        Parameters
        ----------
        tracks_list : List[Track]
            List of Tracks for the sensor manager to observe.
        timestamp: datetime.datetime
            The time for the actions to be produced for.

        Returns
        -------
        Iterable[Tuple[Sensor, Action]]
            The actions and associated sensors produced by the sensor manager."""
        all_action_generators = dict()

        for sensor in self.sensors:
            action_generators = sensor.actions(timestamp)
            all_action_generators[sensor] = action_generators  # set of generators

        def config_from_trial(trial):
            config = defaultdict(list)
            for i, (sensor, generators) in enumerate(all_action_generators.items()):

                for j, generator in enumerate(generators):
                    if isinstance(generator, RealNumberActionGenerator):
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", UserWarning)
                            value = trial.suggest_float(
                                f'{i}{j}', generator.min, generator.max + generator.epsilon,
                                step=getattr(generator, 'resolution', None))
                    else:
                        raise TypeError(f"type {type(generator)} not handled yet")
                    action = generator.action_from_value(value)
                    if action is not None:
                        config[sensor].append(action)
                    else:
                        config[sensor].append(generator.default_action)
            return config

        def optimise_func(trial):
            config = config_from_trial(trial)

            return -self.reward_function(config, tracks, timestamp)

        study = optuna.create_study()
        # will finish study after `timeout` seconds has elapsed.
        study.optimize(optimise_func, n_trials=None, timeout=self.timeout)

        best_params = study.best_params
        config = defaultdict(list)
        for i, (sensor, generators) in enumerate(all_action_generators.items()):
            for j, generator in enumerate(generators):
                if isinstance(generator, RealNumberActionGenerator):
                    action = generator.action_from_value(best_params[f'{i}{j}'])
                else:
                    raise TypeError(f"generator type {type(generator)} not supported")
                if action is not None:
                    config[sensor].append(action)
                else:
                    config[sensor].append(generator.default_action)

        # Return mapping of sensors and chosen actions for sensors
        return [config]
