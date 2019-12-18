# -*- coding: utf-8 -*
import pytest

try:
    from stonesoup.reader.video import VideoClipReader, FFmpegVideoStreamReader
except RuntimeError:
    # Catch FFMPEG import error
    pytest.skip("Failed to import video reader classes. This is possibly "
                "caused due to a missing FFMPEG system executable",
                allow_module_level=True)


def test_video_clip_reader():

    # Expect Type error
    with pytest.raises(TypeError):
        VideoClipReader()

    # TODO: Add more tests


def test_ffmpeg_video_stream_reader():

    # Expect Type error
    with pytest.raises(TypeError):
        FFmpegVideoStreamReader()

    # TODO: Add more tests
