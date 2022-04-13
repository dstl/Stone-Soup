from stonesoup.dataassociator.general import GeneralAssociationGate, OneToOneAssociatorWithGates
from stonesoup.measures import GenericMeasure, Euclidean
from stonesoup.types.state import State


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

    five = 5


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


if __name__ == '__main__':
    words_association_example()
    state_association_example()