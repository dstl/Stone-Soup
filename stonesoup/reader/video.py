# -*- coding: utf-8 -*-
"""Video readers for Stone Soup.

This is a collection of video readers for Stone Soup, allowing quick reading
of video data/streams.
"""

import datetime
import threading
from abc import abstractmethod
from queue import Queue
from typing import Mapping, Tuple, Sequence

import numpy as np
try:
    import ffmpeg
    import moviepy.editor as mpy
except ImportError as error:
    raise ImportError(
        "Usage of video processing classes requires that the optional"
        "package dependencies 'moviepy' and 'ffmpeg-python' are installed. "
        "This can be achieved by running "
        "'python -m pip install stonesoup[video]'")\
        from error


from .base import SensorDataReader
from .file import FileReader
from ..base import Property
from ..buffered_generator import BufferedGenerator
from ..types.sensordata import ImageFrame


class FrameReader(SensorDataReader):
    """FrameReader base class

    A FrameReader produces :class:`~.SensorData` in the form of
    :class:`~ImageFrame` objects.
    """

    @property
    def frame(self):
        return self.sensor_data

    @abstractmethod
    @BufferedGenerator.generator_method
    def frames_gen(self):
        """Returns a generator of frames for each time step.

        Yields
        ------
        : :class:`datetime.datetime`
            Datetime of current time step
        : set of :class:`~.ImageFrame`
            Generated frame in the time step
        """
        raise NotImplementedError

    @BufferedGenerator.generator_method
    def sensor_data_gen(self):
        """Returns a generator of frames for each time step.

        Note
        ----
        This is just a wrapper around (and therefore performs identically
        to) :meth:`~frames_gen`.

        Yields
        ------
        : :class:`datetime.datetime`
            Datetime of current time step
        : set of :class:`~.ImageFrame`
            Generated frame in the time step
        """
        yield from self.frames_gen()


class VideoClipReader(FileReader, FrameReader):
    """VideoClipReader

    A simple reader that uses MoviePy_ to read video frames from a file.

    Usage of MoviePy allows for the application of clip transformations
    and effects, as per the MoviePy documentation_. Upon instantiation,
    the underlying MoviePy `VideoFileClip` instance can be accessed
    through the :attr:`~clip` class property. This can then be used
    as expected, e.g.:

    .. code-block:: python

        # Rearrange RGB to BGR
        def arrange_bgr(image):
            return image[:, :, [2, 1, 0]]

        reader = VideoClipReader("path_to_file")
        reader.clip = reader.clip.fl_image(arrange_bgr)

        for timestamp, frame in reader:
            # The generated frame.pixels will now
            # be arranged in BGR format.
            ...

    .. _MoviePy: https://zulko.github.io/moviepy/index.html
    .. _documentation: https://zulko.github.io/moviepy/getting_started/effects.html
     """  # noqa:E501
    start_time = Property(datetime.timedelta,
                          doc="Start time expressed as duration "
                              "from the start of the clip",
                          default=datetime.timedelta(seconds=0))
    end_time = Property(datetime.timedelta,
                        doc="End time expressed as duration "
                            "from the start of the clip",
                        default=None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        end_time_sec = self.end_time.total_seconds() if self.end_time is not None else None
        self.clip = mpy.VideoFileClip(str(self.path)) \
            .subclip(self.start_time.total_seconds(), end_time_sec)

    @BufferedGenerator.generator_method
    def frames_gen(self):
        start_time = datetime.datetime.now()
        for timestamp_sec, pixels in self.clip.iter_frames(with_times=True):
            timestamp = start_time + datetime.timedelta(seconds=timestamp_sec)
            frame = ImageFrame(pixels, timestamp)
            yield timestamp, frame


class FFmpegVideoStreamReader(FrameReader):
    """ FFmpegVideoStreamReader

    A threaded reader that uses ffmpeg-python_ to read frames from video
    streams (e.g. RTSP) in real-time.


    Notes
    -----
    - Use of this class requires that FFmpeg_ is installed on the host machine.
    - By default, FFmpeg performs internal buffering of frames leading to a \
    slight delay in the incoming frames (0.5-1 sec). To remove the delay it \
    is recommended to set ``input_opts={'threads': 1, 'fflags': 'nobuffer'}`` \
    when instantiating a reader, e.g: .

    .. code-block:: python

        video_reader = FFmpegVideoStreamReader('rtsp://192.168.0.10:554/1/h264minor',
                                               input_opts={'threads': 1, 'fflags': 'nobuffer'})
        for timestamp, frame in video_reader:
            ....

    .. _ffmpeg-python: https://github.com/kkroening/ffmpeg-python
    .. _FFmpeg: https://www.ffmpeg.org/download.html

    """

    input_file: str = Property(
        doc="Input source to read video stream from, passed as input file argument. This can "
            "include any valid FFmpeg input e.g. rtsp URL, device name when using 'dshow'/'v4l2'")
    buffer_size: int = Property(
        default=1,
        doc="Size of the frame buffer. The frame buffer is used to cache frames in cases where "
            "the stream generates frames faster than they are ingested by the reader. If "
            "`buffer_size` is less than or equal to zero, the buffer size is infinite.")
    input_opts: Mapping[str, str] = Property(
        default=None,
        doc="FFmpeg input options, provided in the form of a dictionary, whose keys correspond to "
            "option names. (e.g. ``{'fflags': 'nobuffer'}``). The default is ``{}``.")
    output_opts: Mapping[str, str] = Property(
        default=None,
        doc="FFmpeg output options, provided in the form of a dictionary, whose keys correspond "
            "to option names. The default is ``{'f': 'rawvideo', 'pix_fmt': 'rgb24'}``.")
    filters: Sequence[Tuple[str, Sequence[str], Mapping[str, str]]] = Property(
        default=None,
        doc="FFmpeg filters, provided in the form of a list of filter name, sequence of "
            "arguments, mapping of key/value pairs (e.g. ``[('scale', ('320', '240'), {})]``). "
            "Default `None` where no filter will be applied. Note that :attr:`frame_size` may "
            "need to be set in when video size changed by filter.")
    frame_size: Tuple[int, int] = Property(
        default=None,
        doc="Tuple of frame width and height. Default `None` where it will be detected using "
            "`ffprobe` against the input, but this may yield wrong width/height (e.g. when "
            "filters are applied), and such this option can be used to override.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.input_opts is None:
            self.input_opts = {}
        if self.output_opts is None:
            self.output_opts = {'f': 'rawvideo', 'pix_fmt': 'rgb24'}
        if self.filters is None:
            self.filters = []

        self.buffer = Queue(maxsize=self.buffer_size)

        if self.frame_size is not None:
            self._stream_info = {
                'width': self.frame_size[0],
                'height': self.frame_size[1]}
        else:
            # Probe stream information
            self._stream_info = next(
                s
                for s in ffmpeg.probe(self.input_file, **self.input_opts)['streams']
                if s['codec_type'] == 'video')

        # Initialise stream
        self.stream = ffmpeg.input(self.input_file, **self.input_opts)
        for filter_ in self.filters:
            filter_name, filter_args, filter_kwargs = filter_
            self.stream = self.stream.filter(
                filter_name, *filter_args, **filter_kwargs
            )
        self.stream = (
            self.stream
            .output('pipe:', **self.output_opts)
            .global_args('-y', '-loglevel', 'panic')
            .run_async(pipe_stdout=True)
        )

        # Initialise capture thread
        self._capture_thread = threading.Thread(target=self._run)
        self._capture_thread.daemon = True
        self._capture_thread.start()

    @BufferedGenerator.generator_method
    def frames_gen(self):
        while self._capture_thread.is_alive():
            # if not self.buffer.empty():
            frame = self.buffer.get()
            timestamp = frame.timestamp
            yield timestamp, frame

    def _run(self):
        while self.stream.poll() is None:
            width = int(self._stream_info['width'])
            height = int(self._stream_info['height'])

            # Read bytes from stream
            in_bytes = self.stream.stdout.read(width * height * 3)

            if in_bytes:
                # Transform bytes to pixels
                frame_np = (
                    np.frombuffer(in_bytes, np.uint8)
                    .reshape([height, width, 3])
                )
                frame = ImageFrame(frame_np, datetime.datetime.now())

                # Write new frame to buffer
                self.buffer.put(frame)
