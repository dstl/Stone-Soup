Sensors
=======

.. toctree::
    stonesoup.sensor.radar

.. toctree::
    stonesoup.sensor.action

.. automodule:: stonesoup.sensor
    :no-members:

.. automodule:: stonesoup.sensor.base
    :show-inheritance:

Reference frames and mounted sensors
------------------------------------

Reference-frame conventions vary between navigation, aerospace and sensor systems, so external
orientation values should be converted explicitly before being supplied to Stone Soup rather than
assuming similarly named yaw, pitch and roll values are interchangeable.

For moving platforms, :attr:`~stonesoup.movable.movable.MovingMovable.orientation` is derived from
the platform velocity (direction of motion), with roll taken as zero. A mounted sensor's
``mounting_offset`` is a translation from the platform reference point in the platform-local frame,
while ``rotation_offset`` is applied relative to the platform orientation. Sensor orientation is
exposed through :attr:`~stonesoup.sensor.base.PlatformMountable.orientation`.

Orientation and rotation vectors use rotations about the Cartesian ``x``, ``y`` and ``z`` axes.
When importing values from another convention, check axis order, handedness, rotation sign and
whether the source describes an active vector rotation or a passive change of reference frame.
The API documentation for :attr:`~stonesoup.movable.movable.MovingMovable.orientation` and
:attr:`~stonesoup.sensor.base.PlatformMountable.orientation` defines Stone Soup's exact sign
convention.

.. automodule:: stonesoup.sensor.sensor
    :show-inheritance:


Passive
-------
.. automodule:: stonesoup.sensor.passive
    :show-inheritance:

Categorical
-----------
.. automodule:: stonesoup.sensor.categorical
    :show-inheritance:

Gas
---
.. automodule:: stonesoup.sensor.gas
    :show-inheritance:
