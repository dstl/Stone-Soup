from ..feeder.simple import OriginalStateDetectionSpaceFeeder
from ..tracker.simple import SingleTargetTracker
from ..types.angle import Elevation, Bearing


class AngleSingleTargetTracker(SingleTargetTracker):
    r"""A simple angle-only single target tracker.

    Track a single object using Stone Soup components in platform-centric
    polar angles, elevation and azimuth, and optionally range. It is important to note that the
    state vector is assumed to take the form of

    .. math::
        \mathbf{x}_{k} = \begin{bmatrix}\theta\\\dot{\theta}\\\varphi\\\dot{\varphi}\end{bmatrix}

    or

    .. math::
        \mathbf{x}_{k} =
            \begin{bmatrix}
                \theta\\
                \dot{\theta}\\
                \varphi\\
                \dot{\varphi}\\
                r\\
                \dot{r}
            \end{bmatrix}

    where :math:`\theta` is the elevation, :math:`\varphi` is the bearing, :math:`r` is the range
    to the object and the dot notation is used to define the rate of change of the parameters.
    The Angle-Only tracker works the same way as its Cartesian counterpart by first calling the
    :attr:`data_associator` with the active track, and then either updating the track state with
    the prediction if no detection is associated to the track. The difference is then the relevant
    state vector components are cast to :attr:`Elevation` and :attr:`Bearing` types. The track is
    checked for deletion by the :attr:`deleter`, and if deleted the :attr:`initiator` is called to
    generate a new track. Similarly, if no track is present (i.e. tracker is initialised or
    deleted in previous iteration), only the :attr:`initiator` is called.

    Parameters
    ----------

    Attributes
    ----------
    track : :class:`~.Track`
        Current track being maintained. Also accessible as the sole item in
        :attr:`tracks`
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.detector = OriginalStateDetectionSpaceFeeder(reader=self.detector)

    def __next__(self):
        time, detections = next(self.detector_iter)

        if self._track is not None:
            associations = self.data_associator.associate(
                self.tracks, detections, time)
            if associations[self._track]:
                state_post = self.updater.update(associations[self._track])
                state_post.state_vector[0] = Elevation(state_post.state_vector[0])
                state_post.state_vector[2] = Bearing(state_post.state_vector[2])
                self._track.append(state_post)
            else:
                self._track.append(
                    associations[self._track].prediction)

        if self._track is None or self.deleter.delete_tracks(self.tracks):
            new_tracks = self.initiator.initiate(detections, time)
            if new_tracks:
                self._track = new_tracks.pop()
            else:
                self._track = None

        return time, self.tracks
