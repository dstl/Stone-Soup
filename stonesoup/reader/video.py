# -*- coding: utf-8 -*-
"""Video readers for Stone Soup.

This is a collection of video readers for Stone Soup, allowing quick reading
of video data/streams.
"""

from abc import abstractmethod
import datetime
import numpy as np
import ffmpeg
import moviepy.editor as mpy
import threading
from queue import Queue

from ..base import Property
from ..buffered_generator import BufferedGenerator
from ..types.sensordata import ImageFrame
from .base import SensorDataReader
from .file import FileReader
from .url import UrlReader


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
        to) :py:meth:`~frames_gen`.

        Yields
        ------
        : :class:`datetime.datetime`
            Datetime of current time step
        : set of :class:`~.ImageFrame`
            Generated frame in the time step
        """
        return self.frames_gen()


class VideoClipReader(FileReader, FrameReader):
    """VideoClipReader

    A simple reader that uses MoviePy_ to read video frames from a file.

    Usage of MoviePy allows for the application of clip transformations
    and effects, as per the MoviePy documentation_. Upon instantiation,
    the underlying MoviePy `VideoFileClip` instance can be accessed
    through the :py:attr:`~clip` class property. This can then be used
    as expected, e.g.:

    .. code-block:: python

        # Rearrange RGB to BGR
        def arrange_bgr(image):
            return image[:, :, [2, 1, 0]]

        reader = VideoClipReader("path_to_file")
        reader.clip = reader.clip.fl_image(arrange_rgb)

        for timestamp, frame in reader:
            # The generated frame.pixels will now
            # be arranged in BGR format.
            ...

    .. _MoviePy: https://zulko.github.io/moviepy/index.html
    .. _documentation: https://zulko.github.io/moviepy/getting_started/effects.html
     """
    start_time = Property(datetime.timedelta,
                          doc="Start time expressed as duration "
                              "from the start of the clip",
                          default=0)
    end_time = Property(datetime.timedelta,
                        doc="End time expressed as duration "
                            "from the start of the clip",
                        default=None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.clip = mpy.VideoFileClip(str(self.path)) \
            .subclip(self.start_time.total_seconds(),
                     self.end_time.total_seconds())

    @BufferedGenerator.generator_method
    def frames_gen(self):
        start_time = datetime.datetime.now()
        for timestamp_sec, frame in self.clip.iter_frames(with_times=True):
            timestamp = start_time + datetime.timedelta(seconds=timestamp_sec)
            frame = ImageFrame(frame, timestamp)
            yield timestamp, frame


class FFmpegVideoStreamReader(UrlReader, FrameReader):
    """ FFmpegVideoStreamReader

    A threaded reader that uses ffmpeg-python_ to read frames from video
    streams (e.g. RTSP) in real-time.


    Notes
    -----
    - Use of this class requires that FFmpeg_ is installed on the host machine.
    - By default, FFmpeg performs internal buffering of frames leading to a \
    slight delay in the incoming frames (0.5-1 sec). To remove the delay it is \
    recommended to set ``input_opts={'threads': 1, 'fflags': 'nobuffer'}`` when \
    instantiating a reader, e.g: .

    .. code-block:: python

        video_reader = FFmpegVideoStreamReader('rtsp://192.168.0.10:554/1/h264minor',
                                               input_opts={'threads': 1, 'fflags': 'nobuffer'})
        for timestamp, frame in video_reader:
            ....

    .. _ffmpeg-python: https://github.com/kkroening/ffmpeg-python
    .. _FFmpeg: https://www.ffmpeg.org/download.html

    """

    buffer_size = Property(int,
                           doc="Size of the frame buffer. The frame "
                               "buffer is used to cache frames in cases "
                               "where the stream generates frames faster "
                               "than they are ingested by the reader. "
                               "If `buffer_size` is less than or equal to "
                               "zero, the buffer size is infinite.",
                           default=1)

    input_opts = Property(dict,
                          doc="FFmpeg input options, provided in the form of "
                              "a dictionary, whose keys correspond to option "
                              "names. (e.g. ``{'fflags': 'nobuffer'}``). "
                              "The default is ``{}``.",
                          default={})
    output_opts = Property(dict,
                           doc="FFmpeg output options, provided in the form "
                               "of a dictionary, whose keys correspond to "
                               "option names. The default is "
                               "``{'f': 'rawvideo', 'pix_fmt': 'rgb24'}``.",
                           default={'f': 'rawvideo', 'pix_fmt': 'rgb24'})

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.buffer = Queue(maxsize=self.buffer_size)

        # Initialise stream
        self.stream = (
            ffmpeg
                .input(self.url.geturl(), **self.input_opts)
                .output('pipe:', **self.output_opts)
                .global_args('-y', '-loglevel', 'panic')
                .run_async(pipe_stdout=True)
        )

        # Probe stream information
        self._stream_info = next(
            s for s in ffmpeg.probe(self.url.geturl())['streams'] if s['codec_type'] == 'video')

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
