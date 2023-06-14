from stonesoup.sensor.action import ActionGenerator, Action
from stonesoup.types.state import StateVector


class GridActionGenerator(ActionGenerator):
    """ This generates potential actions in a grid like directions (Up, Down, Left, Right)"""
    def __contains__(self, item):
        return item in iter(self)

    def __iter__(self):
        yield MovePositionAction(generator=self,
                                 end_time=self.end_time,
                                 target_value=self.current_value)
        for dim in range(0, 2):
            for n in range(-15, 20, 5):
                if n == 0:
                    continue
                value = StateVector([0, 0])
                value[dim] += n
                yield MovePositionAction(generator=self,
                                         end_time=self.end_time,
                                         target_value=self.current_value + value)


class MovePositionAction(Action):

    def act(self, current_time, timestamp, init_value):
        return self.target_value
