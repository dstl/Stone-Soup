"""
Generic One to One Association Examples
===============================================
These examples demonstrate how to use the :class:`~.OneToOneAssociator` class for three varied uses.
The three examples include associating States, Tracks and words.
"""

# %%
# First some generic imports and variables are set up as they are needed for all of the examples
import datetime

from stonesoup.dataassociator.general import OneToOneAssociator
from stonesoup.non_state_measures import StateSequenceMeasure, MeanMeasure, SetComparisonMeasure
from stonesoup.dataassociator.tracktotrack import OneToOneTrackAssociator
from stonesoup.measures import GenericMeasure, Euclidean
from stonesoup.plotter import Plotterly
from stonesoup.types.state import State
from stonesoup.types.track import Track

colours = ["darkgreen", "firebrick", "gold", "mediumvioletred", "dodgerblue", "black", "blue",
           "lime"]
# %%
# One to One States Association Example
# ---------------------------------------
# The :class:`~.OneToOneAssociator` can be used for associating states with the right
# :attr:`~.OneToOneAssociator.measure`. Consider the scenario having two conflicting sources
# reporting the location of objects. An associator can be used to judge where multiple sensors are
# observing the same object. For simplicity a standard :class:`~.State` with no uncertainty
# information is used. This means the :class:`~.Euclidean` metric is appropriate to compare the
# states.


# %%
# Create Scenario
# ^^^^^^^^^^^^^^^^

# %%
# We have states from source A and source B marked as `states_from_a` and `states_from_b`
# respectively.
states_from_a = [State([1, 5]), State([0, 5]), State([1, 0]), State([2, 2]), State([4, 3]),
                 State([8, 6])]
states_from_b = [State([0, 3]), State([1, 6]), State([2, 1]), State([3, 0]), State([3, 1]),
                 State([4, 6]), State([2, 6])]

state_names = {**{state: f"state a{idx}" for idx, state in enumerate(states_from_a)},
               **{state: f"state b{idx}" for idx, state in enumerate(states_from_b)}}

# %%
# Visualise the scenario
colours_iter = iter(colours)

plotter = Plotterly()
plotter.plot_tracks(tracks=[Track(state) for state in states_from_a],
                    mapping=[0, 1], track_label="Source A",
                    mode="markers", marker=dict(symbol="cross", color=next(colours_iter)))

plotter.plot_tracks(tracks=[Track(state) for state in states_from_b],
                    mapping=[0, 1], track_label="Source B",
                    mode="markers", marker=dict(symbol="circle", color=next(colours_iter)))

plotter.fig

# %%
# It’s not immediately clear which states should be associated to each other

# %%
# Create Associator & Associate States
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# %%
# **Create Associator**.
# Create an :class:`~.OneToOneAssociator` which can associate :class:`~.State` objects. The
# :class:`~.Euclidean` metric is used to compare the objects.
state_associator = OneToOneAssociator(measure=Euclidean(mapping=[0, 1]),
                                      maximise_measure=False,
                                      association_threshold=3)

# %%
# **Associate States**.
# The :class:`~.OneToOneAssociator` will minimise the total measure, in this case Euclidean
# distance between pairs of objects. For pairs of objects with a measure equal to or above the
# :attr:`~.OneToOneAssociator.threshold`, these pairs won’t be associated together.
associations, unassociated_states_a, unassociated_states_b = \
    state_associator.associate(states_from_a, states_from_b)


# %%
# Results of Association
# ^^^^^^^^^^^^^^^^^^^^^^^
# The results will be visualised and printed

colours_iter = iter(colours)
plotter = Plotterly()
for idx, assoc in enumerate(associations.associations):
    state_from_a = [state for state in assoc.objects if state in states_from_a][0]
    state_from_b = [state for state in assoc.objects if state in states_from_b][0]

    colour = next(colours_iter)
    track = Track(state_from_a, init_metadata=dict(source="a", association=idx))
    plotter.plot_tracks(track,
                        mapping=[0, 1], mode="markers",
                        track_label=f"{state_names[state_from_a]}, Association {idx}",
                        marker=dict(symbol="cross", color=colour))

    track = Track(state_from_b, init_metadata=dict(source="b", association=idx))
    plotter.plot_tracks(track, mapping=[0, 1], mode="markers",
                        track_label=f"{state_names[state_from_b]}, Association {idx}",
                        marker=dict(symbol="circle", color=colour))

    dist_between_states = Euclidean()(state_from_a, state_from_b)
    print(f"State {list(state_from_a.state_vector)} from source A is associated to state "
          f"{list(state_from_b.state_vector)} from source B. The distance between the states is "
          f"{dist_between_states:.1f}")


for state in unassociated_states_a:
    print(f"State {list(state.state_vector)} from source A isn't associated any states from "
          f"source B.")
    colour = next(colours_iter)
    track = Track(state, init_metadata=dict(source="a", association=None))
    plotter.plot_tracks(track, mapping=[0, 1],
                        track_label=f"{state_names[state]}, No Association",
                        mode="markers", marker=dict(symbol="cross", color=colour))

for state in unassociated_states_b:
    print(f"State {list(state.state_vector)} from source B isn't associated any states from "
          f"source A.")
    colour = next(colours_iter)
    track = Track(state, init_metadata=dict(source="b", association=None))
    plotter.plot_tracks(track, mapping=[0, 1],
                        track_label=f"{state_names[state]}, No Association",
                        mode="markers", marker=dict(symbol="circle", color=colour))

plotter.fig

# %%
# **Summary**
#  * Five states from source A have been associated to five states from source B
#  * Three states aren’t associated to another state
#  * State a5 and b5 would be associated but the distance between them is above the threshold
#  * State b0 isn’t associated to a state due to there being better combinations of other states

# %%
# Track to Track Association Example
# -----------------------------------
# This example demonstrates the ability of the :class:`~.OneToOneTrackAssociator` to associate
# tracks together.

# %%
# Create Scenario
# ^^^^^^^^^^^^^^^^
# Six tracks are created and are plotted

start_time = datetime.datetime(2023, 1, 1, 0, 0, 0)

track_a1 = Track(states=[State(state_vector=[[i], [i]],
                               timestamp=start_time + datetime.timedelta(seconds=i))
                         for i in range(10)],
                 id="Track a1")

track_b1 = Track(states=[State(state_vector=[[0], [0]], timestamp=start_time)] +
                        [State(state_vector=[[i + 2], [i + 2.3]],
                               timestamp=start_time + datetime.timedelta(seconds=i))
                         for i in range(1, 7)], id="Track b1")

track_a2 = Track(states=[State(state_vector=[[10-i], [i]],
                               timestamp=start_time + datetime.timedelta(seconds=i))
                         for i in range(10)],
                 id="Track a2")

track_b2 = Track(states=[State(state_vector=[[6 - i/6], [i]],
                               timestamp=start_time + datetime.timedelta(seconds=i))
                         for i in range(10)],
                 id="Track b2")

track_b3 = Track(states=[State(state_vector=[[i+0.1], [i]],
                               timestamp=start_time + datetime.timedelta(seconds=i + 20))
                         for i in range(10)],
                 id="Track b3")

track_a3 = Track(states=[State(state_vector=[[i + 5], [15]],
                               timestamp=start_time + datetime.timedelta(seconds=i))
                         for i in range(10)],
                 id="Track a3")

tracks_a = {track_a1, track_a2, track_a3}
tracks_b = {track_b1, track_b2, track_b3}

# %%
# Plot the tracks
colours_iter = iter(colours)

plotter = Plotterly()

colour = next(colours_iter)
for track in tracks_a:
    plotter.plot_tracks(track, mapping=[0, 1], track_label=track.id, marker=dict(color=colour))

colour = next(colours_iter)
for track in tracks_b:
    plotter.plot_tracks(track, mapping=[0, 1], track_label=track.id, marker=dict(color=colour))

plotter.fig

# %%
# **Track a1** and **Track b1** are close to each other. They should be associated to each other
# unless the tolerance for association is very high.
#
# **Track a2** and **Track b2** are close to each other. Depending on the association threshold they
# may or may not be associated to each other.
#
# **Track a3** is too far away to associated with any of the other tracks.
#
# **Track b3** appears to be closer to `Track a1` than `Track b1` is. However `Track b3` is takes
# place 20 seconds after track `Track a1` and `Track b1`, therefore there are no overlapping time
# periods for association.

# %%
# Create Associator & Associate States
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# %%
# The `full_state_sequence_measure` measure will apply the Euclidean state measure to each state in
# the tracks, with the same time. This produces a multiple measures for each state.
full_state_sequence_measure = StateSequenceMeasure(measure=Euclidean(mapping=[0, 1]))

# %%
# The `track_measure` will take the multiple measures from `full_state_sequence_measure` and
# condense it down into one single measure by taking the mean
track_measure = MeanMeasure(measure=full_state_sequence_measure)

# %%
# The :class:`~.OneToOneTrackAssociator` is a subclass of :class:`~.OneToOneAssociator` and
# :class:`~.TwoTrackToTrackAssociator`.
associator = OneToOneTrackAssociator(measure=track_measure,
                                     association_threshold=5,  # Any pairs >= 5 will be discarded
                                     maximise_measure=False  # The minimum measure is needed
                                     )

associations, unassociated_a, unassociated_b = associator.associate(tracks_a, tracks_b)

# %%
# Results of Association
# ^^^^^^^^^^^^^^^^^^^^^^^
# The results will be visualised and printed
plotter = Plotterly()
colours_iter = iter(colours)

for assoc in associations.associations:
    print('Associated together', [track.id for track in assoc.objects])
    colour = next(colours_iter)
    for track in assoc.objects:
        plotter.plot_tracks(track, mapping=[0, 1], track_label=track.id,
                            marker=dict(color=colour))

print("Not Associated in A: ", [track.id for track in unassociated_a])
print("Not Associated in B: ", [track.id for track in unassociated_b])

for track in [*unassociated_a, *unassociated_b]:
    colour = next(colours_iter)
    plotter.plot_tracks(track, mapping=[0, 1], track_label=track.id, marker=dict(color=colour))

plotter.fig


# %%
# Generic Association Example
# -----------------------------------
# The association algorithm can be to associated many things. In this example we'll associate words
# to demonstrate the versatility of the algorithm

# %%
# Example 1 - Measure 1
# ^^^^^^^^^^^^^^^^^^^^^^^
class WordMeasure(GenericMeasure):
    """
    This a crude measure to compare how similar words are
    This measure calculates the number of letters that both words have in common and then
    divides it by the total number of unique letters
    """
    def __call__(self, word_1: str, word_2: str) -> float:
        return SetComparisonMeasure()(set(word_1.lower()), set(word_2.lower()))


# %%
# An example scenario where an application can only accept standard colour names. However the
# input parameter can accept CCS colours
standard_colours = ["White", "Black", "Yellow", "Red", "Blue", "Green", "Orange", "Purple",
                    "Grey"]

received_colours_scheme = ["FloralWhite", "LightGreen", "Magenta"]


associator = OneToOneAssociator(measure=WordMeasure(),
                                maximise_measure=True,
                                association_threshold=0.3)

association_dict = associator.association_dict(standard_colours, received_colours_scheme)

print("Received Colour:\tAssociated Standard Colour")
for received_colour in received_colours_scheme:
    standard_colour = association_dict[received_colour]
    print(received_colour, "matched with: \t", standard_colour)

# %%
# Magneta shouldn't be match with Orange. We need a better measure


# %%
# Example 1 - Measure 2
# ^^^^^^^^^^^^^^^^^^^^^^^
class BetterWordMeasure(GenericMeasure):
    """
    This measure looks if:
        the words are identical or
        a phrase or word is contained within another word/phrase
    """
    def __call__(self, word_1: str, word_2: str) -> float:
        word_1 = word_1.lower()
        word_2 = word_2.lower()

        if word_1 == word_2:
            return 1.0
        elif word_1 in word_2 or word_2 in word_1:
            return 0.5
        else:
            return 0.0


associator = OneToOneAssociator(measure=BetterWordMeasure(),
                                maximise_measure=True,
                                association_threshold=0.3
                                )

association_dict = associator.association_dict(standard_colours, received_colours_scheme)

print("Received Colour:\tAssociated Standard Colour")
for received_colour in received_colours_scheme:
    standard_colour = association_dict[received_colour]
    print(received_colour, "matched with: \t", standard_colour)


# %%
# Example 2 - Measure 2
# ^^^^^^^^^^^^^^^^^^^^^^^

received_colours_scheme = ["LightSeaGreen", "OrangeRed", "MediumVioletRed"]

association_dict = associator.association_dict(standard_colours, received_colours_scheme)

print("Received Colour:\tAssociated Standard Colour")
for received_colour in received_colours_scheme:
    standard_colour = association_dict[received_colour]
    print(received_colour, "matched with: \t", standard_colour)
