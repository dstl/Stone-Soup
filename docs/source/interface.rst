Component Interfaces
====================

This sections contains the base classes for the different components within the
Stone Soup Framework.

Enabling Components
-------------------
.. autoclass:: stonesoup.detector.Detector
.. autoclass:: stonesoup.feeder.Feeder
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
.. autoclass:: stonesoup.simulator.GroundTruthSimulator
.. autoclass:: stonesoup.simulator.SensorSimulator


Tracker Components
------------------
.. autoclass:: stonesoup.dataassociator.DataAssociator
.. autoclass:: stonesoup.deletor.Deletor
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
