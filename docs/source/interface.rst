Component Interfaces
====================

This sections contains the base classes for the different components within the
Stone Soup Framework.

Enabling Components
-------------------
.. autoclass:: stonesoup.detector.Detector
    :inherited-members:
.. autoclass:: stonesoup.feeder.Feeder
    :inherited-members:
.. autoclass:: stonesoup.metricgenerator.MetricGenerator
.. autoclass:: stonesoup.tracker.Tracker

Data Input
^^^^^^^^^^
.. autoclass:: stonesoup.reader.DetectionReader
.. autoclass:: stonesoup.reader.GroundTruthReader
.. autoclass:: stonesoup.reader.SensorDataReader

Data Output
^^^^^^^^^^^
.. autoclass:: stonesoup.writer.MetricsWriter
.. autoclass:: stonesoup.writer.TrackWriter

Simulators
^^^^^^^^^^
.. autoclass:: stonesoup.simulator.DetectionSimulator
    :inherited-members:
.. autoclass:: stonesoup.simulator.GroundTruthSimulator
    :inherited-members:
.. autoclass:: stonesoup.simulator.SensorSimulator
    :inherited-members:


Tracker Components
------------------
.. autoclass:: stonesoup.dataassociator.DataAssociator
.. autoclass:: stonesoup.deleter.Deleter
.. autoclass:: stonesoup.hypothesiser.Hypothesiser
.. autoclass:: stonesoup.initiator.Initiator
.. autoclass:: stonesoup.mixturereducer.MixtureReducer
.. autoclass:: stonesoup.predictor.Predictor
.. autoclass:: stonesoup.updater.Updater

Models
^^^^^^
.. autoclass:: stonesoup.models.control.ControlModel
    :inherited-members:
.. autoclass:: stonesoup.models.measurement.MeasurementModel
    :inherited-members:
.. autoclass:: stonesoup.models.transition.TransitionModel
    :inherited-members:
