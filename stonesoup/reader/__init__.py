"""Reader classes are used for getting data into the framework."""
from .base import Reader, DetectionReader, GroundTruthReader, SensorDataReader

__all__ = [
    'Reader', 'DetectionReader', 'GroundTruthReader', 'SensorDataReader']
