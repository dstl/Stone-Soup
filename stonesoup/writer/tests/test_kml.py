"""Tests for KML writer classes."""
import os


from ..kml import KMLTrackWriter, CoordinateSystems


def test_kml_trackwriter(ss_tracker_obj, tmpdir,
                         tks_coord_system=CoordinateSystems.LLA,
                         coord_ref_point=(0.0, 0.0, 0.0)):
    """Test for KMLTrackWriter."""
    output_filename = tmpdir.join("ss_output.kml")
    kml_track_writer = KMLTrackWriter(ss_tracker_obj, output_filename,
                                      coordinate_system=tks_coord_system,
                                      reference_point=coord_ref_point)
    kml_track_writer.write()
    # Check if the file was created.
    assert os.path.isfile(output_filename)
