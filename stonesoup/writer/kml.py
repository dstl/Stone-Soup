# -*- coding: utf-8 -*-
from pathlib import Path

from ..base import Property

from ..tracker import Tracker
from .base import Writer, TrackWriter, MetricsWriter

from pykml.parser import Schema

class KMLMetricsWriter(MetricsWriter):
	pass

class KMLTrackWriter(TrackWriter):
	
	def __init__(self):
		pass


class KMLWriter(Writer):
	path = Property(Path,
                    doc="File to save data to. Str will be converted to Path")
    groundtruth_source = Property(GroundTruthReader, default=None)
    sensor_data_source = Property(SensorDataReader, default=None)
    detections_source = Property(DetectionReader, default=None)
    tracks_source = Property(Tracker, default=None)


    def __init__(self, path, *args, **kwargs):
    	if not isinstance(path, Path):
    		path = Path(path)  # Ensure Path
    	super().__init__(path, *args, **kwargs)


    def validate(self, doc):
    	schema_ogc = Scehma("ogckml22.xsd")
    	if (schema_ogc.validate(doc)):
    		return True