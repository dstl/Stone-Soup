"""
This is a helper module for StoneSoup KML writer classes.

It defines low-level definitions and functionality for writing KML files.
Usage:
    kml_obj = SSKML()
    kml_obj.appendTracks(pos_matrix_lla, track_ids, time_matrix)
    kml_obj.appendDetections(pos_array_lla, time_array)
    kml_obj.write(path)

"""
import matplotlib

from enum import Enum
from matplotlib import cm
from simplekml import Kml

# Document Name
DOC_NAME = "Stone Soup Output"

# Folder name for tracks
TRACKS_FOLDER_NAME = 'All Stone Soup Tracks'


# Folder name for  detections
DETECTIONS_FOLDER_NAME = 'All Stone Soup Detections'


# Track Point Normal Icon Scale
TRACK_POINT_NORMAL_ICON_SCALE = "0.5"

# Track Point Normal Icon HREF
TRACK_POINT_NORMAL_ICON_HREF = "http://earth.google.com"\
    "/images/kml-icons/track-directional/track-none.png"

# Track Point Highlight Icon Scale
TRACK_POINT_HIGHLIGHT_ICON_SCALE = "1.2"

# Track Point Highlight Icon HREF
TRACK_POINT_HIGHLIGHT_ICON_HREF = "http://earth.google.com"\
    "/images/kml-icons/track-directional/track-none.png"

# Track line width
TRACK_LINE_WIDTH = "6"

# Detection Normal Icon Scale
DETECTION_NORMAL_ICON_SCALE = "0.5"

# Detection Normal Icon HREF
DETECTION_NORMAL_ICON_HREF = "http://earth.google.com"\
    "/images/kml-icons/track-directional/track-none.png"

# Detection Highlight Icon Scale
DETECTION_HIGHLIGHT_ICON_SCALE = "1.2"

# Detection Highlight Icon HREF
DETECTION_HIGHLIGHT_ICON_HREF = "http://earth.google.com"\
    "/images/kml-icons/track-directional/track-none.png"


class AltitudeModes(Enum):
    """Class that defines altitude modes available in Google Earth."""

    clampToGround = 'clampToGround'
    relativeToGround = 'relativeToGround'
    absolute = 'absolute'


class SSKMLTrack(object):
    """Class that writes SS tracks and detections.

    Used by SSKML() object. Do not call externally.
    """

    def __init__(self, parent_folder, track_id, track_color,
                 tks_position_array, tks_timestamps=None):
        """
        Class constructor.

        Input params:
            i. parent_folder: Parent folder for this track object.

            ii. track_id: Unique track id.

            iii. track_color: Colour to use for track.

            iv. tks_position_array: Array of track posisitions.
                Each element of the array is an array of size 3,
                which consists of latitude, longitude, and altitude
                denoting position.

            v. tks_timestamps: Array of timestamps. Timestamps in this
               array correspond to the posistions in tks_position_array.
               That is, position at tks_posisiton_array[i] corresponds to
               the timestamp at tks_timestamps[i].
        """
        self.track_folder = parent_folder.newfolder(name=track_id)
        # Create time stamp node
        if tks_timestamps:
            time_stamps_sorted = sorted(tks_timestamps)
            time_stamps_start = time_stamps_sorted[0]
            time_stamps_end = time_stamps_sorted[-1]

            self.track_folder.timespan.begin = time_stamps_start.strftime(
                '%Y-%m-%dT%H:%M:%SZ')
            self.track_folder.timespan.end = time_stamps_end.strftime(
                '%Y-%m-%dT%H:%M:%SZ')

        # Points folder
        self.points_folder = self.track_folder.newfolder(name="Points")
        if tks_timestamps and (len(tks_timestamps) == len(tks_position_array)):
            for track_position, timestamp in zip(
                    tks_position_array, tks_timestamps):
                track_position = track_position.ravel()
                # Add point place mark
                self.append_ss_point_placemark(track_position[0],
                                               track_position[1],
                                               track_position[2],
                                               track_color,
                                               timestamp)
        else:
            for track_position in tks_position_array:
                track_position = track_position.ravel()
                # Add point placemark
                self.append_ss_point_placemark(track_position[0],
                                               track_position[1],
                                               track_position[2],
                                               track_color)

        # Track Line
        self.append_ss_line_placemark(tks_position_array, track_color)

    @staticmethod
    def update_placemark_style(simplekml_placemark, color_str, is_track=False):
        """Static method for updating placemark style."""
        if (is_track):
            # Placemark is a track
            simplekml_placemark.stylemap.highlightstyle.iconstyle.scale =\
                TRACK_POINT_HIGHLIGHT_ICON_SCALE
            simplekml_placemark.stylemap.highlightstyle.iconstyle.icon.href =\
                TRACK_POINT_NORMAL_ICON_HREF
            simplekml_placemark.stylemap.highlightstyle.iconstyle.color =\
                color_str
            simplekml_placemark.stylemap.normalstyle.iconstyle.scale =\
                TRACK_POINT_NORMAL_ICON_SCALE
            simplekml_placemark.stylemap.normalstyle.iconstyle.icon.href =\
                TRACK_POINT_HIGHLIGHT_ICON_HREF
            simplekml_placemark.stylemap.normalstyle.iconstyle.color =\
                color_str
            return simplekml_placemark
        simplekml_placemark.stylemap.highlightstyle.iconstyle.scale =\
            DETECTION_HIGHLIGHT_ICON_SCALE
        simplekml_placemark.stylemap.highlightstyle.iconstyle.icon.href =\
            DETECTION_HIGHLIGHT_ICON_HREF
        simplekml_placemark.stylemap.highlightstyle.iconstyle.color =\
            color_str
        simplekml_placemark.stylemap.normalstyle.iconstyle.scale =\
            DETECTION_NORMAL_ICON_SCALE
        simplekml_placemark.stylemap.normalstyle.iconstyle.icon.href =\
            DETECTION_NORMAL_ICON_HREF
        simplekml_placemark.stylemap.normalstyle.iconstyle.color =\
            color_str

        return simplekml_placemark

    def append_ss_point_placemark(
            self, lat, lon, alt, track_color,
            timestamp=None, altitude_mode=AltitudeModes.relativeToGround):
        """Append point placemark."""
        point_placemark = self.points_folder.newpoint()
        if (timestamp):
            point_placemark.timestamp.when = timestamp.strftime(
                '%Y-%m-%dT%H:%M:%SZ')
        point_placemark.extrude = "0"
        point_placemark.altitudemode = altitude_mode.value
        # Add styles
        point_placemark = SSKMLTrack.update_placemark_style(
            point_placemark, track_color, True)
        point_placemark.coords = [[lon, lat, alt]]

    def append_ss_line_placemark(
            self, lla_ordered_list, track_color,
            altitude_mode=AltitudeModes.relativeToGround):
        """Append line placemark."""
        line_placemark = self.track_folder.newlinestring()
        # Add line style
        line_placemark.style.linestyle.color = track_color
        line_placemark.style.linestyle.width = TRACK_LINE_WIDTH

        line_placemark.extrude = "0"
        line_placemark.altitudemode = altitude_mode.value
        # Add line coordinates
        coord_list = [[lla_pnt.ravel()[1], lla_pnt.ravel()[0],
                       lla_pnt.ravel()[2]] for lla_pnt in lla_ordered_list]
        line_placemark.coords = coord_list


class SSKML(object):
    """Class that encapsulates KML object for SS.

    This is the class that should be called externally.
    Usage:
    kml_obj = SSKML()
    kml_obj.appendTracks(pos_matrix_lla, track_ids, time_matrix)
    kml_obj.appendDetections(pos_array_lla, time_array)
    kml_obj.write(path)
    """

    def __init__(self):
        """Class constructor."""
        self._kmlRoot = Kml()
        self._doc = self._kmlRoot.document
        self._doc.name = DOC_NAME
        self._tks_style_idx = 0
        self._det_style_idx = 0
        self._tks_max_colors = 128
        self._det_max_colors = 128
        self._tks_cmap_norm = matplotlib.colors.Normalize(
            vmin=1, vmax=self._tks_max_colors)
        self._det_cmap_norm = matplotlib.colors.Normalize(
            vmin=1, vmax=self._det_max_colors)

    def _get_det_color(self):
        self._det_style_idx += 1
        color_rgba = cm.jet(self._det_cmap_norm(self._det_style_idx))
        color_hex = matplotlib.colors.rgb2hex(color_rgba)
        # Matplotlib HEX -> #RGBA, KML HEX -> #ABGR
        color_hex_kml = "#ff{}".format(color_hex[6:0:-1])
        return color_hex_kml

    def _get_track_color(self):
        self._tks_style_idx += 1
        color_rgba = cm.jet(self._tks_cmap_norm(self._tks_style_idx))
        color_hex = matplotlib.colors.rgb2hex(color_rgba)
        # Matplotlib HEX -> #RGBA, KML HEX -> #ABGR
        color_hex_kml = "#ff{}".format(color_hex[6:0:-1])
        return color_hex_kml

    def append_tracks(self, tks_position_matrix, tks_ids,
                      tks_timestamp_matrix=None):
        """Append tracks to the KML.

        Input params:
             i. tks_position_matrix: A numpy array consisting of
             track positions. Each entry in the array is an array of size 3
             that consists of latitude, longitude, and altitude.

             ii. tks_ids: An array of unique track ids.
             iii. tks_timestamp_matrix (optional): An array of timestamps.
             If provided, this array should be the same size as the array
             tks_timestamp_matrix, and the timestamp at index 'i'
             (i.e. tks_timestamp_matrix[i]) should correspond to the position
             at the same index in tks_positions_matrix
             (i.e. tks_positions_matrix[i]).
        """
        tks_folder = self._doc.newfolder(name=TRACKS_FOLDER_NAME)
        num_tracks = len(tks_position_matrix)
        self._tks_max_colors = num_tracks
        self._tks_cmap_norm = matplotlib.colors.Normalize(
            vmin=1, vmax=self._tks_max_colors)
        if (tks_timestamp_matrix):
            for track_positions, track_id, track_timestamps in zip(
                    tks_position_matrix, tks_ids, tks_timestamp_matrix):
                track_color_str = self._get_track_color()
                SSKMLTrack(tks_folder, track_id, track_color_str,
                           track_positions, track_timestamps)
        else:
            for track_positions, track_id in zip(tks_position_matrix, tks_ids):
                track_color_str = self._get_track_color()
                SSKMLTrack(tks_folder, track_id,
                           track_color_str, track_positions)

    def append_detections(self, det_position_array, det_time_array=None):
        """Append detections to the KML.

        Input params:
            i. det_position_array: An array of positions. Each element in
            the array is an array of size 3, which consists of latitude,
            longitude, and altitude.
            ii. det_time_array (optional): An array of timestamps.
            If provided, this array should be the same size as the array
            det_time_array, and the timestamp at index 'i'
            (i.e. det_time_array[i]) should correspond to the position
            at the same index in det_position_array
            (i.e. det_position_array[i]).
        """
        detections_folder = self._doc.newfolder(
            name=DETECTIONS_FOLDER_NAME)
        num_detections = len(det_position_array)
        self._det_max_colors = num_detections
        self._det_cmap_norm = matplotlib.colors.Normalize(
            vmin=1, vmax=self._det_max_colors)

        if (det_time_array is not None):
            k = 1
            for det_position, det_timestamp in zip(
                    det_position_array, det_time_array):
                det_position = det_position.ravel()
                det_color_str = self._get_det_color()
                det_folder = detections_folder.newfolder(
                    name="Detection_{}".format(k))
                det_placemark = det_folder.newpoint()
                det_placemark.timestamp.when = det_timestamp.strftime(
                    '%Y-%m-%dT%H:%M:%SZ')
                det_placemark.extrude = "0"
                det_placemark.altitudemode =\
                    AltitudeModes.relativeToGround.value
                # Add styles
                det_placemark = SSKMLTrack.update_placemark_style(
                    det_placemark, det_color_str, False)
                # Update coordinates
                det_placemark.coords = [
                    [det_position[1], det_position[0], det_position[2]]]
                k += 1
        else:
            k = 1
            for det_position in det_position_array:
                det_position = det_position.ravel()
                det_color_str = self._get_det_color()
                det_folder = detections_folder.newfolder(
                    name="Detection_{}".format(k))
                det_placemark = det_folder.newpoint()
                det_placemark.extrude = "0"
                det_placemark.altitudemode =\
                    AltitudeModes.relativeToGround.value
                # Add styles
                det_placemark = SSKMLTrack.update_placemark_style(
                    det_placemark, det_color_str, False)
                # Update coordinates
                det_placemark.coords = [
                    [det_position[1], det_position[0], det_position[2]]]
                k += 1

    def write(self, output_path):
        """Write kml at the path given by output_path."""
        self._kmlRoot.save(str(output_path))

    def validate(self):
        """Validate kml using the default schemas included in simplekml."""
        pass
