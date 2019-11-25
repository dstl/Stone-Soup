import cv2
import datetime

from stonesoup.reader.video import VideoClipReader

VIDEO_PATH = r'C:\Users\sglvladi\OneDrive\TensorFlow\datasets\video-samples\sample-9_Trim.mp4'
start_time = datetime.timedelta(seconds=0)
end_time = datetime.timedelta(seconds=20)

# Rearrange RGB to BGR
def arrange_bgr(image):
    return image[:, :, [2, 1, 0]]

video_reader = VideoClipReader(VIDEO_PATH, start_time, end_time)
video_reader.clip = video_reader.clip.fl_image(arrange_bgr)

for timestamp, frame in video_reader:

    print(timestamp)

    # Extract image
    image = frame.pixels

    # Display image
    cv2.imshow('VIDEO', image)
    cv2.waitKey(1)
