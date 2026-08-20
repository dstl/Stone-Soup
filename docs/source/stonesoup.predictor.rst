Predictors
==========

.. automodule:: stonesoup.predictor
    :no-members:

Prediction timing in trackers
-----------------------------

A predictor can evaluate a state at any timestamp supplied to its ``predict`` method. When a
predictor is used inside a tracker, however, prediction times are normally driven by the timestamps
yielded by the tracker's detector. The tracker does not automatically insert extra prediction
states between two detector timestamps.

If denser predicted states are required across a measurement gap, either call the predictor at the
intermediate timestamps explicitly, or have the detector/input generator yield those intermediate
timestamps with an empty detection set. The tracker will then advance existing tracks using their
prediction when no detection is associated at that timestep.

.. automodule:: stonesoup.predictor.base
    :show-inheritance:

Kalman
------

.. automodule:: stonesoup.predictor.kalman
    :show-inheritance:

Particle
--------

.. automodule:: stonesoup.predictor.particle
    :show-inheritance:

Kernel
------

.. automodule:: stonesoup.predictor.kernel
    :show-inheritance:

Ensemble
--------

.. automodule:: stonesoup.predictor.ensemble
    :show-inheritance:

Information
-----------

.. automodule:: stonesoup.predictor.information
    :show-inheritance:

Accumulated State Densities
---------------------------

.. automodule:: stonesoup.predictor.asd
    :show-inheritance:

Categorical
-----------

.. automodule:: stonesoup.predictor.categorical
    :show-inheritance:

Composite
---------

.. automodule:: stonesoup.predictor.composite
    :show-inheritance:
