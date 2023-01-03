from ...base import Property
from ...types.state import State
from ...sensor.action import Action, ActionGenerator


class PlatformAction(Action):
    destination: State = Property()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class PlatformActionGenerator(ActionGenerator):
    owner: object = Property(doc="ActionableMovementController or Platform "
                                 "with ActionableMovementController")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
