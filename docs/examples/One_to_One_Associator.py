"""
One to One Associator Example
===============================================

"""

from stonesoup.dataassociator.general import GeneralAssociationGate, OneToOneAssociatorWithGates, RecentTrackMeasure
from stonesoup.measures import GenericMeasure, Euclidean
from stonesoup.types.state import State

import datetime


def words_association_example():

    class LetterMeasure(GenericMeasure):

        def __call__(self, item1: str, item2: str) -> float:
            r"""
            Compute the distance between a pair of :class:`~.str` objects

            Parameters
            ----------
            item1 : str
            item2 : str

            Returns
            -------
            float
                distance measure between a pair of input objects

            """
            measure = 0
            for letter1 in item1:
                if letter1 in item2:
                    measure = measure + 1

            return measure


    class FirstLetterGate(GeneralAssociationGate):

        def __call__(self, item1: str, item2: str) -> bool:
            return item1[0] == item2[0]

    words1 = ["aaaaa", "bbbab", "abcdef", "ccccc", "zzzzz", "aaa"]
    words2 = ["aaaba", "cdef", "bbc", "cbbc", "ppppp"]

    ga = OneToOneAssociatorWithGates(measure=LetterMeasure(),
                                     maximise_measure=True,
                                     association_threshold=0.1,
                                     gates=[FirstLetterGate()])

    associations, unassociated_a, unassociated_b = ga.associate(words1, words2)

    for assoc in associations.associations:
        print(*assoc.objects)

    print("Not Associated in A: ", unassociated_a)
    print("Not Associated in B: ", unassociated_b)


def state_association_example():
    states1 = [State([1, 5]), State([0, 5]), State([1, 0]), State([2, 2]), State([4, 3])]
    states2 = [State([0, 3]), State([1, 6]), State([2, 1]), State([3, 0]), State([3, 1])]

    ga = OneToOneAssociatorWithGates(measure=Euclidean(mapping=[0, 1]),
                                     maximise_measure=False,
                                     association_threshold=5)

    associations, unassociated_a, unassociated_b = ga.associate(states1, states2)

    for assoc in associations.associations:
        print(*assoc.objects)
        print()

    print("Not Associated in A: ", unassociated_a)
    print("Not Associated in B: ", unassociated_b)


def track_association_example():

    from stonesoup.types.state import State
    from stonesoup.types.track import Track

    start_time = datetime.datetime(2019, 1, 1, 14, 0, 0)

    track_a1 = Track(states=[State(state_vector=[[i], [i]], timestamp=start_time + datetime.timedelta(seconds=i)) for i in range(10)],
                     id="track_a1")

    # 2nd track should be associated with track1 from the second timestamp to
    # the 6th
    track_b1 = Track(states=[State(state_vector=[[0], [0]], timestamp=start_time)] + [
        State(state_vector=[[i + 2], [i + 2.3]],
              timestamp=start_time + datetime.timedelta(seconds=i))
        for i in range(1, 7)], id="track_b1")

    track_a2 = Track(states=[State(state_vector=[[10-i], [i]], timestamp=start_time + datetime.timedelta(seconds=i)) for i in range(10)],
                     id="track_a2")

    track_b2 = Track(states=[State(state_vector=[[6 - i/6], [i]],
                                   timestamp=start_time + datetime.timedelta(seconds=i)) for i in
                             range(10)], id="track_b2")

    # 3rd is at a different time so should not associate with anything
    track_b3 = Track(states=[
        State(state_vector=[[i+0.1], [i]],
              timestamp=start_time + datetime.timedelta(seconds=i + 20))
        for i in range(10)], id="track_b3")

    # 4th is outside the association threshold
    track_a3 = Track(states=[
        State(state_vector=[[i + 5], [15]],
              timestamp=start_time + datetime.timedelta(seconds=i))
        for i in range(10)], id="track_a3")

    tracks_a = {track_a1, track_a2, track_a3}
    tracks_b = {track_b1, track_b2, track_b3}

    from stonesoup.plotter import Plotter
    plotter = Plotter()
    plotter.plot_tracks(tracks_a, mapping=[0, 1], track_label="Tracks A")
    plotter.plot_tracks(tracks_b, mapping=[0, 1], track_label="Tracks B")

    import matplotlib.pyplot as plt
    plt.show()

    associator = OneToOneAssociatorWithGates(
        measure=RecentTrackMeasure(state_measure=Euclidean(mapping=[0, 1])),
        maximise_measure=False,
        association_threshold=10,
        fail_gate_value=1e6
    )

    associations, unassociated_a, unassociated_b = associator.associate(tracks_a, tracks_b)

    for assoc in associations.associations:
        print('Associated together', [track.id for track in assoc.objects])

    print("Not Associated in A: ", [track.id for track in unassociated_a])
    print("Not Associated in B: ", [track.id for track in unassociated_b])

if __name__ == '__main__':
    #words_association_example()
    #state_association_example()
    track_association_example()