import matplotlib
from matplotlib import cm
from enum import Enum
from lxml import etree


from . import kmlSettings as settings

class AltitudeModes(Enum):
    clampToGround = 'clampToGround'
    relativeToGround = 'relativeToGround'
    absolute = 'absolute'

class SSKMLTrack(object):
	def __init__(self, track_id, style_id, tks_position_array, tks_timestamps=None):
		self.trackFolder = etree.Element('Folder')
		nameNode = etree.SubElement(self.trackFolder, "name")
		nameNode.text = track_id
		
		# Create TimeSpan node
		if tks_timestamps:
			timeStampsSorted = sorted(tks_timestamps)
			timeStampStart = timeStampsSorted[0]
			timeStampEnd = timeStampsSorted[-1]

			timeSpanNode = SSKMLTrack.getTimeSpanNode(timeStampStart, timeStampEnd)
			self.trackFolder.append(timeSpanNode)

		# Points folder.
		self.pointsFolder = etree.SubElement(self.trackFolder, "Folder")
		nameNode = etree.SubElement(self.pointsFolder, "name")
		nameNode.text = "Points"
		if tks_timestamps and (len(tks_timestamps) == len(tks_position_array)):
			for track_position, timestamp in zip(tks_position_array, tks_timestamps):
				track_position = track_position.ravel()
				self.pointsFolder.append(SSKMLTrack.getSSPointPlacemark(track_position[0],\
					track_position[1], track_position[2], style_id, timestamp))
		else:
			for track_position in tks_position_array:
				track_position = track_position.ravel()
				self.pointsFolder.append(SSKMLTrack.getSSPointPlacemark(track_position[0],\
					track_position[1], track_position[2], style_id))

		# Track line.
		lineStyleID = "{}_L".format(style_id)
		self.trackFolder.append(SSKMLTrack.getSSLinePlacemark(tks_position_array, lineStyleID))

	def getTrackFolder(self):
		return self.trackFolder

	@staticmethod
	def getTimeSpanNode(start_timestamp, end_timestamp):
		tsNode = etree.Element("TimeStamp")
		beginNode = etree.SubElement(tsNode, "begin")
		beginNode.text = start_timestamp.strftime('%Y-%m-%dT%H:%M:%SZ')
		endNode = etree.SubElement(tsNode, "end")
		endNode.text = end_timestamp.strftime('%Y-%m-%dT%H:%M:%SZ')
		return tsNode

	@staticmethod
	def getSSPointPlacemark(lat, lon, alt, style_id,\
		timestamp=None, altitude_mode=AltitudeModes.relativeToGround):
		pmarkNode = etree.Element("Placemark")
		if (timestamp):
			timestampNode = etree.SubElement(pmarkNode, "TimeStamp")
			tsWhenNode = etree.SubElement(timestampNode, "when")
			tsWhenNode.text = timestamp.strftime('%Y-%m-%dT%H:%M:%SZ')
		styleURLNode = etree.SubElement(pmarkNode, "styleUrl")
		styleURLNode.text = "#{}".format(style_id)
		extrudeNode = etree.SubElement(pmarkNode, "extrude")
		extrudeNode.text = "0"
		altitudeMode = etree.SubElement(pmarkNode, "altitudeMode")
		altitudeMode.text = altitude_mode.value
		pointNode = etree.SubElement(pmarkNode, "Point")
		extrudeNode = etree.SubElement(pointNode, "extrude")
		extrudeNode.text = "0"
		altitudeMode = etree.SubElement(pointNode, "altitudeMode")
		altitudeMode.text = altitude_mode.value
		coordinatesNode = etree.SubElement(pointNode, "coordinates")
		coordinatesNode.text = "{},{},{}".format(lon, lat, alt)
		return pmarkNode

	@staticmethod
	def getSSLinePlacemark(lla_ordered_list, style_id,\
		altitude_mode=AltitudeModes.relativeToGround):
		pmarkNode = etree.Element("Placemark")
		nameNode = etree.SubElement(pmarkNode, "name")
		nameNode.text = "Path"
		styleURLNode = etree.SubElement(pmarkNode, "styleUrl")
		styleURLNode.text = "#{}".format(style_id)
		extrudeNode = etree.SubElement(pmarkNode, "extrude")
		extrudeNode.text = "0"
		altitudeMode = etree.SubElement(pmarkNode, "altitudeMode")
		altitudeMode.text = altitude_mode.value
		lineStringNode = etree.SubElement(pmarkNode, "LineString")
		# tessellateNode = etree.SubElement(lineStringNode, "tessellate")
		# tessellateNode.text = "1"
		extrudeNode = etree.SubElement(lineStringNode, "extrude")
		extrudeNode.text = "0"
		altitudeMode = etree.SubElement(lineStringNode, "altitudeMode")
		altitudeMode.text = altitude_mode.value
		coordinatesNode = etree.SubElement(lineStringNode, "coordinates")
		coordStr = ''.join(["{},{},{} ".format(lla_pnt.ravel()[1], lla_pnt.ravel()[0],\
		 lla_pnt.ravel()[2]) for lla_pnt in lla_ordered_list])
		coordStr = coordStr.rstrip()
		coordinatesNode.text = coordStr
		return pmarkNode


class SSKML(object):

	def __init__(self):
		self._kmlRoot = etree.Element('kml', nsmap=settings.NS_MAP)
		self._kmlDoc = etree.SubElement(self._kmlRoot, "Document")
		docName = etree.SubElement(self._kmlDoc, "name")
		docName.text = settings.DOC_NAME
		self._tksStyleIndex = 0
		self._detStyleIndex = 0
		self._tksMaxColors = 128
		self._detMaxColors = 128
		self._tkscmapNorm = matplotlib.colors.Normalize(vmin=1, vmax=self._tksMaxColors)
		self._detcmapNorm = matplotlib.colors.Normalize(vmin=1, vmax=self._detMaxColors)


	def _appendSSTrackStyleNode(self):
		if (self._tksStyleIndex >= self._tksMaxColors):
			styleIDPre = format("{}_{}".format(settings.KML_TRACK_STYLE_ID_PREFIX, int(self._tksMaxColors*0.5)))
			return styleIDPre

		self._tksStyleIndex += 1
		
		styleIDPre = format("{}_{}".format(settings.KML_TRACK_STYLE_ID_PREFIX, self._tksStyleIndex))
		styleColorRGBA = cm.jet(self._tkscmapNorm(self._tksStyleIndex))
		styleColorHEX = matplotlib.colors.rgb2hex(styleColorRGBA)
		# Matplotlib HEX -> #RGBA, KML HEX -> #ABGR
		styleColorKMLHEX = "#ff{}".format(styleColorHEX[6:0:-1])
		# Generate Style Nodes
		ssKMLStylesStr = settings.STONE_SOUP_TRACK_STYLE_NODES_AS_STR
		ssKMLStylesStr = ssKMLStylesStr.replace(settings.KML_STYLE_ID_PLACEHOLDER, styleIDPre)
		ssKMLStylesStr = ssKMLStylesStr.replace(settings.KML_COLOR_PLACEHOLDER, styleColorKMLHEX)
		ssStyleNodes = etree.fromstring(ssKMLStylesStr)
		styleNodes = ssStyleNodes.getchildren()
		[self._kmlDoc.append(sNode) for sNode in styleNodes]
		return styleIDPre

	def _appendSSDetectionStyleNode(self):
		if (self._detStyleIndex >= self._detMaxColors):
			styleIDPre = format("{}_{}".format(settings.KML_TRACK_STYLE_ID_PREFIX, int(self._detMaxColors*0.5)))
			return styleIDPre

		self._detStyleIndex += 1
		
		styleIDPre = format("{}_{}".format(settings.KML_DETECTION_STYLE_ID_PREFIX, self._detStyleIndex))
		styleColorRGBA = cm.jet(self._detcmapNorm(self._detStyleIndex))
		styleColorHEX = matplotlib.colors.rgb2hex(styleColorRGBA)
		# Matplotlib HEX -> #RGBA, KML HEX -> #ABGR
		styleColorKMLHEX = "#ff{}".format(styleColorHEX[6:0:-1])
		# Generate Style Nodes
		ssKMLStylesStr = settings.STONE_SOUP_DETECTION_STYLE_NODES_AS_STR
		ssKMLStylesStr = ssKMLStylesStr.replace(settings.KML_STYLE_ID_PLACEHOLDER, styleIDPre)
		ssKMLStylesStr = ssKMLStylesStr.replace(settings.KML_COLOR_PLACEHOLDER, styleColorKMLHEX)
		ssStyleNodes = etree.fromstring(ssKMLStylesStr)
		styleNodes = ssStyleNodes.getchildren()
		[self._kmlDoc.append(sNode) for sNode in styleNodes]
		return styleIDPre

	@staticmethod
	def getKMLFolderWithName(name):
		folderNode = etree.Element("Folder")
		nameNode = etree.SubElement(folderNode, "name")
		nameNode.text = name
		return folderNode

	# def _appendSSStyles(self, doc_node, color_count = 5):
	# 	norm = matplotlib.colors.Normalize(vmin=1, vmax=colour_count)
	# 	for k in range(1, colour_count):
	# 		styleIDPre = format("{}_{}".format(settings.KML_STYLE_ID_PREFIX, k))
	# 		styleColorRGBA = cm.jet(norm(k))
	# 		styleColorHEX = matplotlib.colors.rgb2hex(styleColorRGBA)
	# 		# Matplotlib HEX -> #RGBA, KML HEX -> #ABGR
	# 		styleColorKMLHEX = "#ff{}".format(styleColorHEX[6:0:-1])
	# 		# Generate Style Nodes
	# 		ssKMLStylesStr = settings.STONE_SOUP_STYLE_NODES_AS_STR
	# 		ssKMLStylesStr = ssKMLStylesStr.replace(settings.KML_STYLE_ID_PLACEHOLDER, styleIDPre)
	# 		ssKMLStylesStr = ssKMLStylesStr.replace(settings.KML_COLOR_PLACEHOLDER, styleColorKMLHEX)
	# 		ssStyleNodes = etree.fromstring(ssKMLStylesStr)
	# 		styleNodes = ssStyleNodes.getchildren()
	# 		[doc_node.append(sNode) for sNode in styleNodes]



	def _appendFolderToDocNode(self, folder_name="Stone Soup Tracks"):
		tksFolder = etree.SubElement(self._kmlDoc, "Folder")
		tksFolderName = etree.SubElement(tksFolder, "name")
		tksFolderName.text = folder_name
		return tksFolder


	def _ssKMLDocFactory(self, num_tracks=5):
		kmlRoot = etree.Element('kml', nsmap=settings.NS_MAP)
		kmlDoc = etree.SubElement(kmlRoot, "Document")
		docName = etree.SubElement(kmlDoc, "name")
		docName.text = settings.DOC_NAME
		self._numStyles = appendSSStyles(kmlDoc, num_tracks)
		
	def appendTracks(self, tks_position_matrix, tks_ids, tks_timestamp_matrix=None):
		tksFolder = self._appendFolderToDocNode("Stone Soup Tracks")
		numTracks = len(tks_position_matrix)
		self._tksMaxColors = numTracks
		self._tkscmapNorm = matplotlib.colors.Normalize(vmin=1, vmax=self._tksMaxColors)
		if (tks_timestamp_matrix):
			for track_positions, track_id, track_timestamps in zip(tks_position_matrix, tks_ids, tks_timestamp_matrix):
				style_id = self._appendSSTrackStyleNode()
				ssKMLTrack = SSKMLTrack(track_id, style_id, track_positions, track_timestamps)
				tksFolder.append(ssKMLTrack.getTrackFolder())
		else:
			for track_positions, track_id in zip(tks_position_matrix, tks_ids):
				style_id = self._appendSSTrackStyleNode()
				ssKMLTrack = SSKMLTrack(track_id, style_id, track_positions, track_timestamps)
				tksFolder.append(ssKMLTrack.getTrackFolder())
		self._kmlDoc.append(tksFolder)


	def appendDetections(self, det_position_array, det_time_array=None):
		ssDetFolder = self._appendFolderToDocNode("Stone Soup Detections")
		numDetections = len(det_position_array)
		self._detMaxColors = numDetections
		self._detcmapNorm = matplotlib.colors.Normalize(vmin=1, vmax=self._detMaxColors)

		if (det_time_array is not None):
			k = 1
			for det_position, det_timestamp in zip(det_position_array, det_time_array):
				det_position = det_position.ravel()
				style_id = self._appendSSDetectionStyleNode()
				detPntFolder = SSKML.getKMLFolderWithName("Detection_{}".format(k))
				ssKMLDet = SSKMLTrack.getSSPointPlacemark(det_position[0], det_position[1],\
					det_position[2], style_id, det_timestamp)
				detPntFolder.append(ssKMLDet)
				ssDetFolder.append(detPntFolder)
				k += 1
		else:
			k = 1
			for det_position in det_position_array:
				det_position = det_position.ravel()
				style_id = self._appendSSDetectionStyleNode()
				detPntFolder = SSKML.getKMLFolderWithName("Detection_{}".format(k))
				ssKMLDet = SSKMLTrack.getSSPointPlacemark(det_position[0], det_position[1],\
					det_position[2], style_id)
				detPntFolder.append(ssKMLDet)
				ssDetFolder.append(detPntFolder)
				k += 1

	def write(self, output_path):
		# Validate

		# Write output
		et = etree.ElementTree(self._kmlRoot)
		et.write(str(output_path), pretty_print=True, xml_declaration=True, encoding='UTF-8')
		# with open(str(output_path), 'w') as f:
		# 	f.write(etree.tostring(self._kmlRoot, pretty_print = True,
		# 		xml_declaration=True, encoding='UTF-8', method='xml'))
	

	def _validate(self):
		pass


