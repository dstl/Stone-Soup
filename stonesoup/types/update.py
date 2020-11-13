# -*- coding: utf-8 -*-
from typing import Any, Optional

from ..base import Property
from .base import Type
from .hypothesis import Hypothesis
from .state import State, GaussianState, ParticleState, SqrtGaussianState
from .mixture import GaussianMixture


def _from_state(
        cls, state: State, update_type: Optional['Update'] = None, **kwargs: Any) -> 'Update':
    """Return new Update instance of suitable type using existing properties

    Parameters
    ----------
    state: State
        :class:`~.State` to use existing properties from, and identify update type from
    update_type: :class:`~.Update`, optional
        Type to use for update, overriding one from :attr:`class_mapping`.
    \\*\\*kwargs: Mapping
        New property names and associate value for use in newly created update.
    """
    try:
        state_type = next(type_ for type_ in type(state).mro() if type_ in cls.class_mapping)
    except StopIteration:
        raise TypeError(f'{cls.__name__} type not defined for {type(state).__name__}')
    if update_type is None:
        update_type = cls.class_mapping[state_type]

    # Use current state kwargs that also properties of update type
    new_kwargs = {
        name: getattr(state, name)
        for name in state_type.properties.keys() & update_type.properties.keys()}
    # And replace them with any newly defined kwargs
    new_kwargs.update(kwargs)

    # Special case for SqrtGaussian
    if issubclass(state_type, SqrtGaussianState):
        new_kwargs['sqrt_covar'] = new_kwargs.pop('covar')

    return update_type(**new_kwargs)


class Update(Type):
    """ Update type

    The base update class. Updates are returned by :class:'~.Updater' objects
    and contain the information that was used to perform the updating"""

    hypothesis: Hypothesis = Property(doc="Hypothesis used for updating")

    class_mapping = {}
    from_state = classmethod(_from_state)

    def __init_subclass__(cls, **kwargs):
        state_type = cls.__bases__[-1]
        Update.class_mapping[state_type] = cls
        super().__init_subclass__(**kwargs)


class StateUpdate(Update, State):
    """ StateUpdate type

    Most simple state update type, where everything only has time
    and a state vector. Requires a prior state that was updated,
    and the hypothesis used to update the prior.
    """


class GaussianStateUpdate(Update, GaussianState):
    """ GaussianStateUpdate type

    This is a simple Gaussian state update object, which, as the name
    suggests, is described by a Gaussian distribution.
    """


class SqrtGaussianStateUpdate(Update, SqrtGaussianState):
    """ SqrtGaussianStateUpdate type

    This is equivalent to a Gaussian state update object, but with the
    covariance of the Gaussian distribution stored in matrix square root
    form.
    """


class GaussianMixtureUpdate(Update, GaussianMixture):
    """ GaussianMixtureUpdate type

    This is a Gaussian mixture update object, which, as the name
    suggests, is described by a Gaussian mixture.
    """


class ParticleStateUpdate(Update, ParticleState):
    """ParticleStateUpdate type

    This is a simple Particle state update object.
    """
