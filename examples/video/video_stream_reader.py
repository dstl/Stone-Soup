import cv2

from stonesoup.reader.video import FFmpegVideoStreamReader

STREAM_URL = 'rtsp://192.168.0.10:554/1/h264minor'
# The following options ensure real-time streaming
input_opts = {'threads': 1, 'fflags': 'nobuffer'}
video_reader = FFmpegVideoStreamReader(STREAM_URL, input_opts=input_opts)

for timestamp, frame in video_reader:

    print(timestamp)

    # Extract image
    image = frame.pixels

    # Display image
    cv2.imshow('VIDEO', image)
    cv2.waitKey(1)
