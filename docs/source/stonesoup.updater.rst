Updaters
========

.. automodule:: stonesoup.updater
    :no-members:

.. automodule:: stonesoup.updater.base
    :show-inheritance:

Kalman
------

.. note::
    :class:`~stonesoup.updater.kalman.KalmanUpdater` requires a linear measurement model that
    provides a :meth:`~stonesoup.models.base.LinearModel.matrix` method. Non-linear measurement
    models, such as bearing-range models, should instead be used with an
    :class:`~stonesoup.updater.kalman.ExtendedKalmanUpdater` or
    :class:`~stonesoup.updater.kalman.UnscentedKalmanUpdater`. To use a standard Kalman filter,
    measurements must be represented by a compatible linear model, for example Cartesian position
    measurements.

.. automodule:: stonesoup.updater.kalman
    :show-inheritance:

Particle
--------

.. automodule:: stonesoup.updater.particle
    :show-inheritance:

Kernel
------

.. automodule:: stonesoup.updater.kernel
    :show-inheritance:

Ensemble
--------

.. automodule:: stonesoup.updater.ensemble
    :show-inheritance:

Recursive
---------

.. automodule:: stonesoup.updater.recursive
    :show-inheritance:

Iterated
--------

.. automodule:: stonesoup.updater.iterated
    :show-inheritance:

Information
-----------

.. automodule:: stonesoup.updater.information
    :show-inheritance:

Accumulated State Densities
---------------------------

.. automodule:: stonesoup.updater.asd
    :show-inheritance:

Point Process
-------------

.. automodule:: stonesoup.updater.pointprocess
    :show-inheritance:

AlphaBeta
---------

.. automodule:: stonesoup.updater.alphabeta
    :show-inheritance:

Sliding Innovation Filter
-------------------------

.. automodule:: stonesoup.updater.slidinginnovation
    :show-inheritance:

Categorical
-----------

.. automodule:: stonesoup.updater.categorical
    :show-inheritance:

Composite
---------

.. automodule:: stonesoup.updater.composite
    :show-inheritance:

Chernoff
--------

.. automodule:: stonesoup.updater.chernoff

Probabilistic
-------------

.. automodule:: stonesoup.updater.probability
    :show-inheritance:
