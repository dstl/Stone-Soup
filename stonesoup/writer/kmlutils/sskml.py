from enum import Enum

import matplotlib
from matplotlib import cm

from simplekml import Kml

from . import kmlSettings


class AltitudeModes(Enum):
    clampToGround = 'clampToGround'
    relativeToGround = 'relativeToGround'
    absolute = 'absolute'


class SSKMLTrack(object):

    def __init__(self, parent_folder, track_id, track_color,
                 tks_position_array, tks_timestamps=None):

        self.track_folder = parent_folder.newfolder(name=track_id)
        # Ceate time stamp node
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
        if (is_track):
            # Placemark is a track
            simplekml_placemark.stylemap.highlightstyle.iconstyle.scale =\
                kmlSettings.TRACK_POINT_HIGHLIGHT_ICON_SCALE
            simplekml_placemark.stylemap.highlightstyle.iconstyle.icon.href =\
                kmlSettings.TRACK_POINT_NORMAL_ICON_HREF
            simplekml_placemark.stylemap.highlightstyle.iconstyle.color =\
                color_str
            simplekml_placemark.stylemap.normalstyle.iconstyle.scale =\
                kmlSettings.TRACK_POINT_NORMAL_ICON_SCALE
            simplekml_placemark.stylemap.normalstyle.iconstyle.icon.href =\
                kmlSettings.TRACK_POINT_HIGHLIGHT_ICON_HREF
            simplekml_placemark.stylemap.normalstyle.iconstyle.color =\
                color_str
            return simplekml_placemark
        simplekml_placemark.stylemap.highlightstyle.iconstyle.scale =\
            kmlSettings.DETECTION_HIGHLIGHT_ICON_SCALE
        simplekml_placemark.stylemap.highlightstyle.iconstyle.icon.href =\
            kmlSettings.DETECTION_HIGHLIGHT_ICON_HREF
        simplekml_placemark.stylemap.highlightstyle.iconstyle.color =\
            color_str
        simplekml_placemark.stylemap.normalstyle.iconstyle.scale =\
            kmlSettings.DETECTION_NORMAL_ICON_SCALE
        simplekml_placemark.stylemap.normalstyle.iconstyle.icon.href =\
            kmlSettings.DETECTION_NORMAL_ICON_HREF
        simplekml_placemark.stylemap.normalstyle.iconstyle.color =\
            color_str

        return simplekml_placemark

    def append_ss_point_placemark(
            self, lat, lon, alt, track_color,
            timestamp=None, altitude_mode=AltitudeModes.relativeToGround):
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
        line_placemark = self.track_folder.newlinestring()
        # Add line style
        line_placemark.style.linestyle.color = track_color
        line_placemark.style.linestyle.width = kmlSettings.TRACK_LINE_WIDTH

        line_placemark.extrude = "0"
        line_placemark.altitudemode = altitude_mode.value
        # Add line coordinates
        coord_list = [[lla_pnt.ravel()[1], lla_pnt.ravel()[0],
                       lla_pnt.ravel()[2]] for lla_pnt in lla_ordered_list]
        line_placemark.coords = coord_list


class SSKML(object):

    def __init__(self):
        self._kmlRoot = Kml()
        self._doc = self._kmlRoot.document
        self._doc.name = kmlSettings.DOC_NAME
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
        tks_folder = self._doc.newfolder(name=kmlSettings.TRACKS_FOLDER_NAME)
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
        detections_folder = self._doc.newfolder(
            name=kmlSettings.DETECTIONS_FOLDER_NAME)
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
        self._kmlRoot.save(str(output_path))

    def validate(self):
        pass
