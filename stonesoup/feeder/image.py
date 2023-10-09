import numpy as np

try:
    import cv2
except ImportError as error:
    raise ImportError('Use of the image feeder classes requires that opencv-python is installed.')\
        from error

from .base import Feeder
from ..base import Property
from ..buffered_generator import BufferedGenerator
from ..types.sensordata import ImageFrame


class CFAR(Feeder):
    """Cell-averaging (CA) Constant False Alarm Rate (CFAR) image data feeder

    The CFAR feeder reads grayscale frames from an appropriate :class:`~.FrameReader`or
    :class:`~.Feeder` and outputs binary frames whose pixel values are either 0 or 255,
    indicating the lack or presence of a detection, respectively.

    See `here <https://en.wikipedia.org/wiki/Constant_false_alarm_rate#Cell-averaging_CFAR>`__ for
    more information on CA-CFAR.

    .. note::
        The frames forwarded by the :attr:`~.CFAR.reader` must be grayscale :class:`~.ImageFrame`
        objects. As such :attr:`~.ImageFrame.pixels` for all frames must be 2-D arrays, containing
        grayscale intensity values.
    """
    train_size: int = Property(doc="The number of train pixels", default=10)
    guard_size: int = Property(doc="The number of guard pixels", default=4)
    alpha: float = Property(doc="The threshold value", default=1.)
    squared: bool = Property(doc="If set to True, the threshold will be computed as a function of "
                                 "the sum of squares. The default is False, in which case a "
                                 "simple sum will be evaluated.", default=False)

    @BufferedGenerator.generator_method
    def data_gen(self):
        for timestamp, frame in self.reader:
            img = frame.pixels.copy()
            output_img = self.cfar(img, self.train_size, self.guard_size, self.alpha, self.squared)
            new_frame = ImageFrame(output_img, frame.timestamp)
            yield timestamp, new_frame

    @staticmethod
    def cfar(input_img, train_size=10, guard_size=4, alpha=1., squared=False):
        """ Perform Constant False Alarm Rate (CFAR) detection on an input image

        Parameters
        ----------
        input_img: numpy.ndarray
            The input grayscale image.
        train_size: int
            The number of train pixels.
        guard_size: int
            The number of guard pixels.
        alpha: float
            The threshold value.
        squared: bool
            If set to True, the threshold will be computed as a function of the sum of squares.
            The default is False, in which case a simple sum will be evaluated.
        Returns
        -------
        numpy.ndarray
            Output image containing 255 for pixels where a target is detected and 0 otherwise.
        """
        # Get width and height of image
        width, height = input_img.shape
        # Compute the CFAR window size
        window_size = 1 + 2*guard_size + 2*train_size
        # Initialise empty output image
        output_img = np.zeros(input_img.shape, np.uint8)
        # Iterate through all pixels
        for i in range(height-window_size):
            for j in range(width-window_size):
                # Compute coordinates of test pixel
                c_i = i + guard_size + train_size
                c_j = j + guard_size + train_size
                # Select the pixels inside the window
                v = input_img[i:i + window_size, j:j + window_size].copy()
                # Exclude pixels inside guard zone
                v[train_size:train_size + 2 * guard_size + 1,
                  train_size:train_size + 2 * guard_size + 1] = 0
                # # The above should be equivalent to the code below
                # v = np.zeros((window_size, window_size))
                # for k in range(window_size):
                #     for l in range(window_size):
                #         if (k >= train_size) and (k < (window_size - train_size)) \
                #                 and (l >= train_size) and (l < (window_size - train_size)):
                #             continue
                #         v[k, l] += input_img[i+k,j+l]
                # Compute the threshold
                if squared:
                    v = v**2
                threshold = np.sum(v) / (window_size**2 - (2*guard_size + 1)**2)
                # Populate the output image
                input_value = input_img[c_i, c_j]
                if squared:
                    input_value = input_value**2
                if input_value/threshold > alpha:
                    output_img[c_i, c_j] = 255
        return output_img


class CCL(Feeder):
    """Connected Component Labelling (CCL) image data feeder

    The CCL feeder reads binary frames from an appropriate :class:`~.FrameReader`or
    :class:`~.Feeder` and outputs labelled frames whose pixel values contain the label of the
    connected component to which each pixel is has been assigned.

    See `here <https://en.wikipedia.org/wiki/Connected-component_labeling#Graphical_example_of_
    two-pass_algorithm>`__ for more information on and example applications of CCL.

    .. note::
        The frames forwarded by the :attr:`~.CCL.reader` must be binary :class:`~.ImageFrame`
        objects. As such :attr:`~.ImageFrame.pixels` for all frames must be 2-D arrays, where each
        element can take only 1 of 2 possible values (e.g. [0 or 1], [0 or 255], etc.).
    """
    @BufferedGenerator.generator_method
    def data_gen(self):
        for timestamp, frame in self.reader:
            img = frame.pixels.copy()
            _, labels_img = cv2.connectedComponents(img)
            new_frame = ImageFrame(labels_img, frame.timestamp)
            yield timestamp, new_frame
