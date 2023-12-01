"""
Generic Association Examples
===============================================
These examples demonstrate how to use the :class:`~.GreedyAssociator` class and the
:class:`~.OneToOneAssociator` class for varied uses. The examples include associating
:class:`~.State`, :class:`~.Track`, words and numbers.
"""

# %%
# Both associators are used for associating objects with an appropriate
# :class:`~.BaseMeasure`. Subclasses of :class:`~.BaseMeasure` take two objects and return
# a float value to assess the separation between the two objects. This is used by the
# :class:`~.OneToOneAssociator` to optimally associate objects.

# %%
# First some generic imports and variables are set up as they are needed for all the examples.
import datetime
import plotly

from stonesoup.dataassociator.general import OneToOneAssociator, GreedyAssociator
from stonesoup.measures.multi import StateSequenceMeasure, MeanMeasure
from stonesoup.measures.base import BaseMeasure, SetComparisonMeasure
from stonesoup.dataassociator.tracktotrack import OneToOneTrackAssociator
from stonesoup.measures import Euclidean
from stonesoup.plotter import Plotterly
from stonesoup.types.state import State
from stonesoup.types.track import Track

colours = plotly.colors.qualitative.T10

# %%
# One-to-One States Association
# ---------------------------------------
# In this example the :class:`~.OneToOneAssociator` is used to associate :class:`~.State` objects.
# Consider the scenario having two conflicting sources reporting the location of objects. An
# associator can be used to judge where multiple sensors are observing the same object. For
# simplicity a standard :class:`~.State` with no uncertainty information is used. This means the
# :class:`~.Euclidean` metric is appropriate to compare the states.


# %%
# Create States
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
# Next, use Plotly to visualise the scenario, with source A states shown in green crosses and
# source B states shown in red circles:
colours_iter = iter(colours)

plotter1 = Plotterly()
plotter1.plot_tracks(tracks=[Track(state) for state in states_from_a],
                     mapping=[0, 1], track_label="Source A",
                     mode="markers", marker=dict(symbol="cross", color=next(colours_iter)))

plotter1.plot_tracks(tracks=[Track(state) for state in states_from_b],
                     mapping=[0, 1], track_label="Source B",
                     mode="markers", marker=dict(symbol="circle", color=next(colours_iter)))

plotter1.fig

# %%
# This scenario has been designed such the optimal association between `states_from_a`
# and `states_from_b` is unclear to the human eye.

# %%
# Create Associator & Associate States
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# %%
# **Create Associator**.
# Create an :class:`~.OneToOneAssociator` which can associate :class:`~.State` objects. The
# :class:`~.Euclidean` metric is used to compare the objects.
state_1_2_1_associator = OneToOneAssociator(measure=Euclidean(mapping=[0, 1]),
                                            maximise_measure=False,
                                            association_threshold=3)

# %%
# **Associate States**. The :class:`~.OneToOneAssociator` will minimise the total measure
# (:class:`~.Euclidean` distance) between the two states. The :class:`~.OneToOneAssociator` uses
# SciPy's :func:`~.scipy.optimize.linear_sum_assignment` function (a modified Jonker-Volgenant
# algorithm) to minimise the distance. For pairs of objects with a distance equal to or above the
# threshold, these pairs won’t be associated together.
#
associations_1_2_1, unassociated_states_a, unassociated_states_b = \
    state_1_2_1_associator.associate(states_from_a, states_from_b)


# %%
# Results of State Association
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The results are visualised below using Plotly:

def plot_state_associations(assocs, unassoc_states_a, unassoc_states_b):

    colors_iter = iter(colours)
    this_plotter = Plotterly()
    for idx, association in enumerate(assocs.associations):
        state_from_a = [state for state in association.objects if state in states_from_a][0]
        state_from_b = [state for state in association.objects if state in states_from_b][0]

        color = next(colors_iter)
        plotting_track = Track(state_from_a, init_metadata=dict(source="a", association=idx))
        this_plotter.plot_tracks(plotting_track,
                                 mapping=[0, 1], mode="markers",
                                 track_label=f"{state_names[state_from_a]}, Association {idx}",
                                 marker=dict(symbol="cross", color=color))

        plotting_track = Track(state_from_b, init_metadata=dict(source="b", association=idx))
        this_plotter.plot_tracks(plotting_track, mapping=[0, 1], mode="markers",
                                 track_label=f"{state_names[state_from_b]}, Association {idx}",
                                 marker=dict(symbol="circle", color=color))

        plotting_track = Track([state_from_a, state_from_b], init_metadata=dict(association=idx))
        this_plotter.plot_tracks(plotting_track, mapping=[0, 1], mode="lines",
                                 track_label=f"Association {idx}",
                                 line=dict(color=color))

        dist_between_states = Euclidean()(state_from_a, state_from_b)
        print(f"State {list(state_from_a.state_vector)} from source A is associated to state "
              f"{list(state_from_b.state_vector)} from source B. The distance between the states "
              f"is {dist_between_states:.1f}")

    for state in unassoc_states_a:
        print(f"State {list(state.state_vector)} from source A isn't associated any states from "
              f"source B.")
        color = next(colors_iter)
        plotting_track = Track(state, init_metadata=dict(source="a", association=None))
        this_plotter.plot_tracks(plotting_track, mapping=[0, 1],
                                 track_label=f"{state_names[state]}, No Association",
                                 mode="markers", marker=dict(symbol="cross", color=color))

    for state in unassoc_states_b:
        print(f"State {list(state.state_vector)} from source B isn't associated any states from "
              f"source A.")
        color = next(colors_iter)
        plotting_track = Track(state, init_metadata=dict(source="b", association=None))
        this_plotter.plot_tracks(plotting_track, mapping=[0, 1],
                                 track_label=f"{state_names[state]}, No Association",
                                 mode="markers", marker=dict(symbol="circle", color=color))

    return this_plotter


plotter2 = plot_state_associations(associations_1_2_1,
                                   unassociated_states_a, unassociated_states_b)
# %%
# The plot below shows the states. Source A states are shown with crosses and source B states are
# shown with circles. Associations are shown by matching colours and lines between the states.

plotter2.fig

# %%
# **Summary**
#  * Five states from source A have been associated to five states from source B
#  * Three states aren’t associated to another state
#  * State a5 and b5 would be associated but the distance between them is above the threshold
#  * State b0 isn’t associated to a state due to there being better combinations of other states

# %%
# Greedy States Association
# ---------------------------------------
# The :class:`~.GreedyAssociator` and the :class:`~.OneToOneAssociator` both inherit from the same
# base class :class:`~.GeneralAssociator`. The same input parameters are used. While the
# :class:`~.OneToOneAssociator` looks for the best global associations the
# :class:`~.GreedyAssociator` only considers the best association for each object one at a time.

# %%
# This example uses the same input parameters as were used in the previous example. The
# :class:`~.Euclidean` metric distance is minimised and the association threshold is 3.

greedy_state_associator = GreedyAssociator(measure=Euclidean(mapping=[0, 1]),
                                           maximise_measure=False,
                                           association_threshold=3)

# %%
# Associate States
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# For each object in the first argument `objects_a` the greedy associator will find the object in
# the second argument `objects_b` with the lowest or higher measure (lowest in this case). It does
# not consider deconflicting ‘b’ objects. As a result ‘b’ objects may be associated multiple times.
associations_greedy_a, unassociated_states_a, unassociated_states_b = \
    greedy_state_associator.associate(states_from_a, states_from_b)

plotter3 = plot_state_associations(associations_greedy_a,
                                   unassociated_states_a, unassociated_states_b)
plotter3.fig

# %%
# **Association Results**
#  * Every 'a' state is associated apart from a5 which is too far from any 'b' state
#  * Four 'b' states aren’t associated to an 'a' state
#  * b1 and b2 are associated to multiple 'a' states

# %%
# Reverse Association Priority
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# In the example above, for each ‘a’ state the best state from ‘b’ was associated to it. In this
# example ‘b’ states have precedence; for each ‘b’ state the best state from ‘a’ was associated to
# it.
associations_greedy_b, unassociated_states_a, unassociated_states_b = \
    greedy_state_associator.associate(states_from_b, states_from_a)

plotter4 = plot_state_associations(associations_greedy_b,
                                   unassociated_states_a, unassociated_states_b)
plotter4.fig

# %%
# **Association Results**
#  * Every 'b' state is associated once
#  * Every 'a' state is associated apart from a5 which is too far from any 'b' state
#  * a0 and a3 are associated to multiple 'a' states


# %%
# Compare States Associations
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The association results from the `OneToOneAssociator` and the two outputs from the
# `GreedyAssociation` are visualised and compared.

def plot_state_associations_compare(this_plotter, assocs, colors_iter, name: str):

    state_pairs_tracks = [Track(association.objects, init_metadata=dict(source=name))
                          for association in assocs.associations]

    this_plotter.plot_tracks(state_pairs_tracks, mapping=[0, 1], mode="lines",
                             track_label=name,
                             line=dict(color=next(colors_iter)))

    return this_plotter


plotter1 = plot_state_associations_compare(plotter1, associations_1_2_1, colours_iter,
                                           name="OneToOneAssociator")
plotter1 = plot_state_associations_compare(plotter1, associations_greedy_a, colours_iter,
                                           name="GreedyAssociation - A")
plotter1 = plot_state_associations_compare(plotter1, associations_greedy_b, colours_iter,
                                           name="GreedyAssociation - B")
plotter1.fig

# %%
# The associations from Greedy A and Greedy B are very different. Therefore the order of input
# arguments for the `GreedyAssociation` are significant.


# %%
# 1-2-1 Track to Track Association
# --------------------------------
# This example demonstrates the ability of the :class:`~.OneToOneTrackAssociator` to associate
# tracks together. This can be used in Track to Track Fusion.

# %%
# Create Tracks
# ^^^^^^^^^^^^^^^^
# Six tracks are created that represent three tracks from each source ‘A’ and ‘B’. These tracks
# represent a varied scenario for a track association algorithm.

start_time = datetime.datetime(2023, 1, 1, 0, 0, 0)

track_a1 = Track(states=[State(state_vector=[[i], [i]],
                               timestamp=start_time + datetime.timedelta(seconds=i))
                         for i in range(10)],
                 id="Track a1")

track_b1 = Track(states=[State(state_vector=[[i + 0], [i + 1]],
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

track_b3 = Track(states=[State(state_vector=[[i+0.5], [i]],
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
# The tracks are plotted. As before, we use different colours to separate `tracks_a` from
# `tracks_b`.
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
# **Track a2** and **Track b2** are close to each other. Depending on the association threshold
# they may or may not be associated to each other.
#
# **Track a3** is too far away to associated with any of the other tracks.
#
# **Track b3** appears to be closer to `Track a1` than `Track b1` is. However, `Track b3` takes
# place 20 seconds after `Track a1` and `Track b1`, therefore there are no overlapping time
# periods for association.

# %%
# Create Associator & Associate Tracks
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# %%
# The `full_state_sequence_measure` (:class:`~.StateSequenceMeasure`) measure will apply the
# Euclidean state measure to each state in the tracks, with the same time. This produces a multiple
# measures for each state.
full_state_sequence_measure = StateSequenceMeasure(state_measure=Euclidean(mapping=[0, 1]))

# %%
# `track_measure` (:class:`~.MeanMeasure`) will take the multiple measures from
# `full_state_sequence_measure` and condense it down into one single measure by taking the mean.
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
# Results of Track Association
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The results will be visualised and printed.
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
# 1-2-1 Word Association
# -----------------------------------
# The association algorithm can be used to associate many things. In this example we'll associate
# words to demonstrate the versatility of the algorithm.

# %%
# In the example scenario an application can only accept standard colour names. However, the user
# interface also accepts CCS colours. The association algorithm  must try to match the user's
# input to a 'standard' colour. The 'standard' colours the application can use are:
standard_colours = ["White", "Black", "Yellow", "Red", "Blue", "Green", "Orange", "Purple",
                    "Grey"]


# %%
# Measure 1 - Example 1
# ^^^^^^^^^^^^^^^^^^^^^^^
# :class:`WordMeasure` is a crude measure to compare how similar words are. It calculates the
# number of letters that both words have in common and divides by the total number of
# unique letters.
class WordMeasure(BaseMeasure):
    def __call__(self, word_1: str, word_2: str) -> float:
        return SetComparisonMeasure()(set(word_1.lower()), set(word_2.lower()))


# %%
# The colours that are inputted by a user:
received_colours_scheme = ["FloralWhite", "LightGreen", "Magenta"]

# %%
# The association process:
associator = OneToOneAssociator(measure=WordMeasure(),
                                maximise_measure=True,
                                association_threshold=0.3)

association_dict = associator.association_dict(standard_colours, received_colours_scheme)

# %%
# Print association results

print("Received Colour:\tAssociated Standard Colour")
for received_colour in received_colours_scheme:
    standard_colour = association_dict[received_colour]
    print(received_colour, "matched with: \t", standard_colour)

# %%
# Magneta shouldn't be match with Orange. We need a better measure.


# %%
# Measure 2 - Example 1
# ^^^^^^^^^^^^^^^^^^^^^^^
# :class:`MatchingWordMeasure` looks for words that are identical or is a word/phrase is
# contained within another word/phrase.
class MatchingWordMeasure(BaseMeasure):
    PERFECT_MATCH = 1.0
    PARTIAL_MATCH = 0.5
    NO_MATCH = 0.0

    def __call__(self, word_1: str, word_2: str) -> float:
        word_1 = word_1.lower()
        word_2 = word_2.lower()

        if word_1 == word_2:
            return self.PERFECT_MATCH
        elif word_1 in word_2 or word_2 in word_1:
            return self.PARTIAL_MATCH
        else:
            return self.NO_MATCH


# %%
# The association process:
associator = OneToOneAssociator(measure=MatchingWordMeasure(),
                                maximise_measure=True,
                                association_threshold=0.3  # Just below PARTIAL_MATCH
                                )

association_dict = associator.association_dict(standard_colours, received_colours_scheme)

# %%
# Print association results:

print("Received Colour:\tAssociated Standard Colour")
for received_colour in received_colours_scheme:
    standard_colour = association_dict[received_colour]
    print(received_colour, "matched with: \t", standard_colour)


# %%
# Measure 2 - Example 2
# ^^^^^^^^^^^^^^^^^^^^^^^

received_colours_scheme = ["LightSeaGreen", "OrangeRed", "MediumVioletRed"]

association_dict = associator.association_dict(standard_colours, received_colours_scheme)

# %%
# Print association results:

print("Received Colour:\tAssociated Standard Colour")
for received_colour in received_colours_scheme:
    standard_colour = association_dict[received_colour]
    print(received_colour, "matched with: \t", standard_colour)


# %%
# Using OneToOneAssociator with an Association Threshold
# -------------------------------------------------------
# The :class:`~.OneToOneAssociator` uses :func:`~scipy.optimize.linear_sum_assignment` which is
# optimal when there is no association requirement. It is no longer optimal when some associations
# may be restricted. This section demonstrates how the
# :attr:`~.GeneralAssociator.association_threshold` and
# :attr:`~.OneToOneAssociator.non_association_cost` attributes effect the output of the
# :class:`~.OneToOneAssociator`.

# %%
# Options for :attr:`~.OneToOneAssociator.non_association_cost`:
#
#  a. ``nan`` - If the goal is to maximise the number of associations
#  b. ``None`` - If you want to simply cut off associations that don’t reach the threshold use
#  c. ``0`` - To maximise the total measure
#  d. ``:attr:`~.GeneralAssociator.association_threshold` ± <small_number> - To minimise the total
#     measure

# %%
# To avoid :func:`~scipy.optimize.linear_sum_assignment` choosing associations that won't pass the
# threshold, a high :attr:`~.OneToOneAssociator.non_association_cost` will be used instead of the
# value of the measure function.

# %%
# **Note**: although infinity is used as the `non_association_cost`, `linear_sum_assignment` cannot
# cope with very large numbers (because very large number + 1 == very large number). Very large
# `non_association_cost` are reduced internally by the `OneToOneAssociator` class before being
# passed to the `linear_sum_assignment` function. See
# :meth:`OneToOneAssociator.individual_weighting` function for detail.

# %%
# This function print the results of the association output.
def print_results(assocs, unassocs_a, unassocs_b, measure):
    assoc_dict = {association.objects[0]: association.objects[1]
                  for association in assocs}

    print(assoc_dict, "\n", unassocs_a, "\n", unassocs_b)

    total = sum(measure(*an_assoc.objects) for an_assoc in assocs.associations)
    print(f"Total Measure={total}")


# %%
# Scenario 1
# ^^^^^^^^^^
# The goal is to minimise the absolute difference between numbers


# %%
# The measure is the ``NumericDifference`` (the absolute difference between the numbers).
class NumericDifference(BaseMeasure):
    def __call__(self, item1: float, item2: float) -> float:
        """ Returns absolute difference between two floats"""
        if item1 == item2:
            return float('nan')
        else:
            return abs(item1-item2)


# %%
# There are two groups of numbers that we want to associate together, ``numbers_a`` and
# ``numbers_b``.
numbers_a = (0, 4, 9)
numbers_b = (3, 8, 50)
association_threshold = 10
measure = NumericDifference()
maximise_measure=False


# %%
# No Association Threshold
# """""""""""""""""""""""""""""""""""""
associator = OneToOneAssociator(measure=measure,
                                maximise_measure=maximise_measure,
                                association_threshold=None)

associations, unassociated_a, unassociated_b = associator.associate(numbers_a, numbers_b)
print_results(associations, unassociated_a, unassociated_b, measure=measure)

# %%
# Every number is associated to another number. The distance `9` and `50` is too large. So an
# association threshold of `10` is applied.

# %%
# Scenario 1 - Option B
# """""""""""""""""""""
associator = OneToOneAssociator(measure=measure,
                                maximise_measure=maximise_measure,
                                association_threshold=association_threshold,
                                non_association_cost=None)

associations, unassociated_a, unassociated_b = associator.associate(numbers_a, numbers_b)
print_results(associations, unassociated_a, unassociated_b, measure=measure)

# %%
# These results are not optimal. The total cost would be lower if `3` was associated with `4` and
# `8` was associated with `9`.

# %%
# Scenario 1 - Option A
# """""""""""""""""""""
associator = OneToOneAssociator(measure=measure,
                                maximise_measure=maximise_measure,
                                association_threshold=association_threshold,
                                non_association_cost=float('nan'))

associations, unassociated_a, unassociated_b = associator.associate(numbers_a, numbers_b)
print_results(associations, unassociated_a, unassociated_b, measure=measure)

# %%
# Scenario 1 - Option C
# """""""""""""""""""""
associator = OneToOneAssociator(measure=measure,
                                maximise_measure=maximise_measure,
                                association_threshold=association_threshold,
                                non_association_cost=0)

associations, unassociated_a, unassociated_b = associator.associate(numbers_a, numbers_b)
print_results(associations, unassociated_a, unassociated_b, measure=measure)

# %%
# Scenario 1 - Option D
# """""""""""""""""""""
if maximise_measure:
    non_association_cost = association_threshold + 0.01
else:
    non_association_cost = association_threshold - 0.01
associator = OneToOneAssociator(measure=measure,
                                maximise_measure=maximise_measure,
                                association_threshold=association_threshold,
                                non_association_cost=non_association_cost)

associations, unassociated_a, unassociated_b = associator.associate(numbers_a, numbers_b)
print_results(associations, unassociated_a, unassociated_b, measure=measure)


# %%
# Scenario 2
# ^^^^^^^^^^
#
class SquareNumeric(BaseMeasure):
    def __call__(self, item1: float, item2: float) -> float:
        if item1 == item2:
            return float('nan')
        return item1*item2


# %%
# There are two groups of numbers that we want to associate together, ``numbers_a`` and
# ``numbers_b``.
numbers_a = (1, 2, 3, 5)
numbers_b = (1, 2, 4, 7)
association_threshold = 5
measure = SquareNumeric()
maximise_measure = True


# %%
# No Association Threshold
# """""""""""""""""""""""""""""""""""""
associator = OneToOneAssociator(measure=measure,
                                maximise_measure=maximise_measure,
                                association_threshold=None)

associations, unassociated_a, unassociated_b = associator.associate(numbers_a, numbers_b)
print_results(associations, unassociated_a, unassociated_b, measure=measure)

# %%
# Every number is associated to another number. The distance `9` and `50` is too large. So an
# association threshold of `10` is applied.

# %%
# Option B
# """""""""""""""""""""
associator = OneToOneAssociator(measure=measure,
                                maximise_measure=maximise_measure,
                                association_threshold=association_threshold,
                                non_association_cost=None)

associations, unassociated_a, unassociated_b = associator.associate(numbers_a, numbers_b)
print_results(associations, unassociated_a, unassociated_b, measure=measure)

# %%
# These results are not optimal. The total cost would be lower if `3` was associated with `4` and
# `8` was associated with `9`.

# %%
# Option A
# """""""""""""""""""""
associator = OneToOneAssociator(measure=measure,
                                maximise_measure=maximise_measure,
                                association_threshold=association_threshold,
                                non_association_cost=float('nan'))

associations, unassociated_a, unassociated_b = associator.associate(numbers_a, numbers_b)
print_results(associations, unassociated_a, unassociated_b, measure=measure)

# %%
# Option C
# """""""""""""""""""""
associator = OneToOneAssociator(measure=measure,
                                maximise_measure=maximise_measure,
                                association_threshold=association_threshold,
                                non_association_cost=0)

associations, unassociated_a, unassociated_b = associator.associate(numbers_a, numbers_b)
print_results(associations, unassociated_a, unassociated_b, measure=measure)

# %%
# Option D
# """""""""""""""""""""
if maximise_measure:
    non_association_cost = association_threshold + 0.01
else:
    non_association_cost = association_threshold - 0.01
associator = OneToOneAssociator(measure=measure,
                                maximise_measure=maximise_measure,
                                association_threshold=association_threshold,
                                non_association_cost=non_association_cost)

associations, unassociated_a, unassociated_b = associator.associate(numbers_a, numbers_b)
print_results(associations, unassociated_a, unassociated_b, measure=measure)

# %%
# Scenario 3
# ^^^^^^^^^^
#
class SquareNumeric(BaseMeasure):
    def __call__(self, item1: float, item2: float) -> float:
        if item1 == item2:
            return float('nan')
        return item1*item2


numbers_a = (1, 2, 3, 6, 10)
numbers_b = (1.1, 1.2, 1.3, 2.2, 7)
association_threshold = 6.95
measure = SquareNumeric()
maximise_measure = False


# %%
# No Association Threshold
# """""""""""""""""""""""""""""""""""""
associator = OneToOneAssociator(measure=measure,
                                maximise_measure=maximise_measure,
                                association_threshold=None)

associations, unassociated_a, unassociated_b = associator.associate(numbers_a, numbers_b)
print_results(associations, unassociated_a, unassociated_b, measure=measure)

# %%
# Every number is associated to another number. The distance `9` and `50` is too large. So an
# association threshold of `10` is applied.

# %%
# Option B
# """""""""""""""""""""
associator = OneToOneAssociator(measure=measure,
                                maximise_measure=maximise_measure,
                                association_threshold=association_threshold,
                                non_association_cost=None)

associations, unassociated_a, unassociated_b = associator.associate(numbers_a, numbers_b)
print_results(associations, unassociated_a, unassociated_b, measure=measure)

# %%
# These results are not optimal. The total cost would be lower if `3` was associated with `4` and
# `8` was associated with `9`.

# %%
# Option A
# """""""""""""""""""""
associator = OneToOneAssociator(measure=measure,
                                maximise_measure=maximise_measure,
                                association_threshold=association_threshold,
                                non_association_cost=float('nan'))

associations, unassociated_a, unassociated_b = associator.associate(numbers_a, numbers_b)
print_results(associations, unassociated_a, unassociated_b, measure=measure)

# %%
# Option C
# """""""""""""""""""""
associator = OneToOneAssociator(measure=measure,
                                maximise_measure=maximise_measure,
                                association_threshold=association_threshold,
                                non_association_cost=0)

associations, unassociated_a, unassociated_b = associator.associate(numbers_a, numbers_b)
print_results(associations, unassociated_a, unassociated_b, measure=measure)

# %%
# Option D
# """""""""""""""""""""
if maximise_measure:
    non_association_cost = association_threshold - 0.01
else:
    non_association_cost = association_threshold + 0.01
associator = OneToOneAssociator(measure=measure,
                                maximise_measure=maximise_measure,
                                association_threshold=association_threshold,
                                non_association_cost=non_association_cost)

associations, unassociated_a, unassociated_b = associator.associate(numbers_a, numbers_b)
print_results(associations, unassociated_a, unassociated_b, measure=measure)

#exit()

# %%
# todo - talk about results

# %%
# Scenario 2
# ^^^^^^^^^^
# In this scenario the measure is the product of the values in an association and the goal is to
# maximise the measure. The numbers from source 'a' are (1, 2, 3, 5) and from source 'b' are
# (1, 2, 4, 7).
#
#
# **Warning**: In this example there is the possibility of associating an object to the same
# object. For example ``1`` could be associated with ``1``. Because the
# :attr:`.Association.objects` attribute is a :class:`set`, this would result in
# :attr:`.Association.objects` only containing the object once. This would break the logic in the
# `print_results` function if it was to occur.


class SquareNumeric(BaseMeasure):
    def __call__(self, item1: float, item2: float) -> float:
        return item1*item2


numbers_a = (1, 2, 3, 5)
numbers_b = (1, 2, 4, 7)


# %%
# Example 1
# """"""""""""""""""""""""""""""""""""""""""""""""""""
associator = OneToOneAssociator(measure=SquareNumeric(),
                                maximise_measure=True,
                                association_threshold=5,
                                non_association_cost=None)

associations, unassociated_a, unassociated_b = associator.associate(numbers_a, numbers_b)
print_results(associations, unassociated_a, unassociated_b, measure=SquareNumeric())
# %%
# Without the threshold `2` would associate with `2` and `1` would associate with `1`. The measure
# output for these associations would be `4` and `1` which is below the threshold `5`. Therefore
# they are not associated.

# %%
# Example 2
# """"""""""""""""""""""""""""""""""""""""""""""""""""
# There are unassociated numbers in example 1 that could be associated. For example `2` could be
# associated with `3`. If the priority is to associate all numbers the
# :attr:`~.OneToOneAssociator.non_association_cost` should be adjusted. It can be changed to a low
# or negative number to influence the algorithm to avoid combinations that don’t meet the
# threshold. ``nan`` can also be used to indicate an incompatible association. Internally the
# `OneToOneAssociator` will replace nan give a large negative value (when maximising the measure).
associator = OneToOneAssociator(measure=SquareNumeric(),
                                maximise_measure=True,
                                association_threshold=5,
                                non_association_cost=float('nan'))

associations, unassociated_a, unassociated_b = associator.associate(numbers_a, numbers_b)
print_results(associations, unassociated_a, unassociated_b, measure=SquareNumeric())
# %%
# Now all the numbers are associated but the total measure is significantly lower (worse).

# %%
# Example 3
# """"""""""""""""""""""""""""""""""""""""""""""""""""
# In this example the :attr:`~.OneToOneAssociator.non_association_cost` is set to `0`. Unassociated
# numbers do not count to the total measure therefore their cost should be `0`, if there are no
# negative measure values.
associator = OneToOneAssociator(measure=SquareNumeric(),
                                maximise_measure=True,
                                association_threshold=5,
                                non_association_cost=0)

associations, unassociated_a, unassociated_b = associator.associate(numbers_a, numbers_b)
print_results(associations, unassociated_a, unassociated_b, measure=SquareNumeric())
# %%
# This is the largest total measure for this scenario.


# %%
# Scenario 4
# ^^^^^^^^^^
# In this scenario the same :attr:`~.GeneralAssociator.measure` is used (`SquareNumeric`) however
# the goal is to minimise the total measure. The input numbers and association threshold are also
# different.
#
# When trying to minimise the measureif the measure is always positive and the goal is to minimise
# the measure, a measure of ``0`` can be achieved by not associating anything. This isn't a useful
# result.

numbers_a = (1, 2, 3, 6, 10)
numbers_b = (1.1, 1.2, 1.3, 2.2, 7)
association_threshold = 6.95

# %%
# No Association Threshold
# """"""""""""""""""""""""""""""""""""""""""""""""""""
# First demonstrate the total without a threshold.
associator = OneToOneAssociator(measure=SquareNumeric(),
                                maximise_measure=False,
                                association_threshold=None,
                                non_association_cost=None)

associations, unassociated_a, unassociated_b = associator.associate(numbers_a, numbers_b)
print_results(associations, unassociated_a, unassociated_b, measure=SquareNumeric())

# %%
# Simple Cut Off
# """"""""""""""""""""""""""""""""""""""""""""""""""""
# This should take the previous result and cut off any associations with a higher measure than the
# ``association_threshold`` (`6.95`).
associator = OneToOneAssociator(measure=SquareNumeric(),
                                maximise_measure=False,
                                association_threshold=association_threshold,
                                non_association_cost=None)

associations, unassociated_a, unassociated_b = associator.associate(numbers_a, numbers_b)
print_results(associations, unassociated_a, unassociated_b, measure=SquareNumeric())
# %%
# Three associations have been cut from the results: 1 & 7, 6 & 1.2 and 10 & 1.1. There is now
# six unassociated numbers.

# %%
# Prioritise Matching
# """"""""""""""""""""""""""""""""""""""""""""""""""""
# To encourage more associations the :attr:`~.OneToOneAssociator.non_association_cost` is set
# to ``nan``. Within :class:`.OneToOneAssociator` ``nan`` is converted to a large number, see
# :attr:`~.OneToOneAssociator.measure_fail_value` for more detail.
associator = OneToOneAssociator(measure=SquareNumeric(),
                                maximise_measure=False,
                                association_threshold=association_threshold,
                                non_association_cost=float('nan'))

associations, unassociated_a, unassociated_b = associator.associate(numbers_a, numbers_b)
print_results(associations, unassociated_a, unassociated_b, measure=SquareNumeric())

# %%
# The total measure is now higher but more numbers are associated. This output contains the lowest
# measure with the most numbers associated.

# %%
# Prioritise Low Score
# """"""""""""""""""""""""""""""""""""""""""""""""""""
# Any associations that don't meet the threshold won't be used regardless of how close they are to
# the threshold. For this example the :attr:`~.OneToOneAssociator.non_association_cost` is set to
# :attr:`~.GeneralAssociator.association_threshold` plus a small number. The small addition is
# important to distinguish it from valid associations.
associator = OneToOneAssociator(measure=SquareNumeric(),
                                maximise_measure=False,
                                association_threshold=association_threshold,
                                non_association_cost=association_threshold+0.1)

associations, unassociated_a, unassociated_b = associator.associate(numbers_a, numbers_b)
print_results(associations, unassociated_a, unassociated_b, measure=SquareNumeric())

# %%
# This result gives a good mix of a low measure with most numbers being associated.


# %%
# Guidance for Setting the Non-Association Cost
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# These are some rules of thumb for setting the :attr:`~.OneToOneAssociator.non_association_cost`
# in the :class:`OneToOneAssociator`:
#
#  * If you want to simply cut off associations that don’t reach the threshold use
#    `non_association_cost=` ``None``.
#  * If the goal is to maximise the number of associations use `non_association_cost=` ``nan``.
#  * To maximise the total measure use `non_association_cost=0`.
#  * To minimise the total measure use `non_association_cost=`
#    :attr:`~.GeneralAssociator.association_threshold` plus a small number.


# %%
# **Summary**
#
# The :class:`~.OneToOneAssociator` and :class:`~.GreedyAssociator` can be used for multiple
# varied purposes. The `OneToOneAssociator` was created originally for track association but can
# be used to associate anything. The examples above show its use in associating
# :class:`~.StateMutualSequence`, :class:`~.State`, :class:`str` and :class:`float`.
# It’s a flexible association class that can be tailored for many use cases.
