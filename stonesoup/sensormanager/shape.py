import numpy as np

from stonesoup.base import Base, Property


class Area(Base):
    """
    Defines a rectangular area in 2D space, defined by minimum and maximum x and y coordinates.
    """
    xmin: float = Property(default=-np.inf, doc="Minimum x coordinate of the area")
    xmax: float = Property(default=np.inf, doc="Maximum x coordinate of the area")
    ymin: float = Property(default=-np.inf, doc="Minimum y coordinate of the area")
    ymax: float = Property(default=np.inf, doc="Maximum y coordinate of the area")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.xmin >= self.xmax or self.ymin >= self.ymax:
            raise ValueError("Area must have xmin < xmax and ymin < ymax")


class AreaOfInterest(Area):
    """
    Defines an area of interest in 2D space, with an associated interest level and access level.
    """
    interest: int = Property(default=1,
                             doc="Interest level of the area."
                                 "0: Low interest, 10: high interest.")
    access: int = Property(default=1,
                           doc="Access level of the area."
                               "0: full access, 10: high threat/no go area.")
