Component Interfaces
====================

This sections contains the base classes for the different components within the
Stone Soup Framework.

Enabling Components
-------------------
.. autoclass:: stonesoup.detector.Detector
    :inherited-members:
    :noindex:
.. autoclass:: stonesoup.feeder.Feeder
    :inherited-members:
    :noindex:
.. autoclass:: stonesoup.metricgenerator.MetricGenerator
    :noindex:
.. autoclass:: stonesoup.smoother.Smoother
    :noindex:
.. autoclass:: stonesoup.tracker.Tracker
    :noindex:

Data Input
^^^^^^^^^^
.. autoclass:: stonesoup.reader.DetectionReader
    :noindex:
.. autoclass:: stonesoup.reader.GroundTruthReader
    :noindex:
.. autoclass:: stonesoup.reader.SensorDataReader
    :noindex:

Data Output
^^^^^^^^^^^
.. autoclass:: stonesoup.writer.MetricsWriter
    :noindex:
.. autoclass:: stonesoup.writer.TrackWriter
    :noindex:

Simulation
^^^^^^^^^^
.. autoclass:: stonesoup.simulator.DetectionSimulator
    :inherited-members:
    :noindex:
.. autoclass:: stonesoup.simulator.GroundTruthSimulator
    :inherited-members:
    :noindex:
.. autoclass:: stonesoup.simulator.SensorSimulator
    :inherited-members:
    :noindex:
.. autoclass:: stonesoup.platform.Platform
    :inherited-members:
    :noindex:
.. autoclass:: stonesoup.sensor.sensor.Sensor
    :inherited-members:
    :noindex:


Algorithm Components
--------------------
.. autoclass:: stonesoup.dataassociator.DataAssociator
    :noindex:
.. autoclass:: stonesoup.deleter.Deleter
    :noindex:
.. autoclass:: stonesoup.hypothesiser.Hypothesiser
    :noindex:
.. autoclass:: stonesoup.gater.Gater
    :noindex:
.. autoclass:: stonesoup.initiator.Initiator
    :noindex:
.. autoclass:: stonesoup.mixturereducer.MixtureReducer
    :noindex:
.. autoclass:: stonesoup.predictor.Predictor
    :noindex:
.. autoclass:: stonesoup.resampler.Resampler
    :noindex:
.. autoclass:: stonesoup.updater.Updater
    :noindex:

Models
^^^^^^
.. autoclass:: stonesoup.models.control.ControlModel
    :inherited-members:
    :noindex:
.. autoclass:: stonesoup.models.measurement.MeasurementModel
    :inherited-members:
    :noindex:
.. autoclass:: stonesoup.models.transition.TransitionModel
    :inherited-members:
    :noindex:
