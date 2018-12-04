import os
import numpy as np
import lxml.etree as etree

from enum import Enum
from datetime import datetime, date, time, timedelta
from pathlib import Path

from .aisutils.decoder import NMEADecoder, InvalidMessage
from .aisutils.parser import MiscParser, MSSISParser, EAParser
from .aisutils.fields import AISField


from .base import DetectionReader
from ..base import Property
from ..types.detection import Detection
from ..types.state import StateVector


READER_DIR, NMEA_FILE = os.path.split(__file__)

AIS_UTILS_DIR = "aisutils"
AIS_TYPES_XML = os.path.join(READER_DIR, AIS_UTILS_DIR, "aistypes.xml")

class NMEASources(Enum):
	MISC = 1
	MSSIS = 2
	ExactEarth = 3


class NMEAReader(DetectionReader):
	bbox = Property(
		(float, float, float, float),
		default=None,
		doc="Optional coordinates for the bounding box. If this is given then only \
		messages from vessels within the bounding box are read as detections. If this is not given then \
		all messages are read. Format: left, bottom, right, top in decimal degrees (latitude/longitude).")
	start_time = Property(
		datetime,
		doc="Time to begin reading NMEA messages.")
	end_time = Property(
		datetime,
		doc="Time to end reading NMEA messages.")

	path = Property(Path, doc="NMEA file path. Str will be converted to Path")

	reference_utc = Property(datetime, default=datetime.combine(date(1970,1,1),time(0,0,0)),
		doc="Optional reference UTC time. Default is 01-01-1970 00:00:00.")

	ais_types_defs = Property(Path, default=AIS_TYPES_XML,
		doc="Optional path to XML file containing 'definitions' of AIS types.")


	src = Property(NMEASources, default=NMEASources.MISC,
		doc="Different providers may organise NMEA messages\
		differently, which may require a different parser.")


	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		if (type(self.src) is not NMEASources):
			raise TypeError("Invalid NMEA source provided.")
		if (self.start_time > self.end_time):
			raise ValueError("End time precedes start time.")


		if not isinstance(self.path, Path):
			self.path = Path(self.path) # Ensure path.

		# Initialize parser and decoder.
		target_time = [self.start_time, self.end_time]
		# Convert to POSIX time.
		time_span = [int((ktime - self.reference_utc).total_seconds()) for ktime in target_time]

		if (self.bbox):
			self.bbox_geom = [[self.bbox[3], self.bbox[0]],
						[self.bbox[3], self.bbox[2]],
						[self.bbox[1], self.bbox[2]],
						[self.bbox[1], self.bbox[0]],
						[self.bbox[3], self.bbox[0]]]
		else:
			self.bbox_geom = None

		self.decoder = NMEADecoder()

		if(self.src == NMEASources.MSSIS):
			self._aLog = MSSISParser(filepath=self.path, time_span=time_span)
		elif(self.src == NMEASources.ExactEarth):
			self._aLog = EAParser(filepath=self.path, time_span=time_span)
		else:
			self._aLog = MiscParser(filepath=self.path, time_span=time_span)

		self._detections = set()


	def loadAISTypes(self):
		parser = etree.XMLParser(remove_blank_text=True)
		try:
			ais_types_xml = etree.parse(self.ais_types_defs, parser)
			root = ais_types_xml.getroot()
			types_list = root.findall(".//aisMessage")
		except Exception as e:
			print("Unable to parse AIS types definition file at {}".format(self.ais_types_defs))
			raise Exception(e)

		ais_fields = [[AISField(field) for field in ais_type] for ais_type in types_list]
		return ais_fields

	def bboxContains(self, lat, lon):
		if (self.bbox is None):
			return True

		counter = 0
		xinters = 0.0
		p1 = [self.bbox_geom[0]]
		N = len(self.bbox_geom)
		for i in range(1, N+1):
			p2 = geom[(i%N)]
			if (lat > min(p1[0], p2[0])):
				if (lat <= max(p1[0], p2[0])):
					if (lon <= max(p1[1], p2[1])):
						if (p1[0] != p2[0]):
							xinters = ((lat - p1[0])*((p2[1] - p1[1])/(p2[0] - p1[0]))) + p1[1]
							if (p1[1] == p2[1] or lon <= xinters):
								counter += 1
			p1 = p2[:]
			if ((count % 2) == 0 or counter == 0):
				return False
			else:
				return True

	def detections(self):
		return self._detections.copy()

	def detections_gen(self):
		ship_mmsi = []
		hits = []
		for record in self._aLog:
			fields = record[5]
			self._detections = set()
			try:
				ts_time = int(record[-1])
				utc_time = self.reference_utc + timedelta(seconds=ts_time)
			except ValueError:
				continue
			msgType = self.decoder.joinFields(fields[0:1], [0x3F], [0])
			self.decoder.setPayload(fields)
			try:
				ais_dict = {fld.name: self.decoder.decodeField(fld) for fld in AISTypes[msgType-1]}
			except IndexError:
				print("Message type {} not defined. Skipping message.".format(msgType))
				continue
			except InvalidMessage as e:
				print("Found an invalid message {}. Skipping message.".format(e.value))
				continue
			# Message type 8.
			if (msgType == 8):
				print("Found message type 8. Skipping.")
				continue

			# Message type is 1, 2, or 3.
			if (msgType < 4):
				if (('Latitude' in ais_dict) and ('Longitude' in ais_dict)):
					coords =[float(ais_dict['Latitude'][0]), float(ais_dict['Longitude'][0])]
					# Check if the coordinate is inside the polygon.
					if (self.bboxContains(coords[0], coords[1])):
						# Append to detections.
						msg_meta = {
								'mmsi': ais_dict['MMSI'][0],
								'cog': ais_dict['CourseOverGround'][0],
								'nav_stat':ais_dict['NavigationStatus'][0]
								}
						det =  Detection(StateVector([coords[0]], [coords[1]]),
							timestamp=utc_time,
							metadata=msg_meta)
						self._detections.add(det)
				else:
					print("Message type 1,2, or 3 found without latitude and longitude. Skipping.")
					continue
			yield time, self.detections
		self._aLog.close()

