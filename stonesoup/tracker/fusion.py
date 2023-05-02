from .base import Tracker

class MultiTargetFusionTracker(Tracker):
    """Takes Detections and Tracks, and fuses them in the style of MultiTargetTracker"""