import math
from typing import Set, List

import numpy as np
from matplotlib.path import Path
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from shapely.geometry.base import BaseGeometry
from vector3d.vector import Vector

from stonesoup.types.angle import Angle
from stonesoup.types.state import ParticleState
from stonesoup.types.track import Track


class CameraCalculator:
    """Ported and modified from https://gist.github.com/luipir/dc33864b53cf6634f9cdd2bce712d3d9"""

    @staticmethod
    def getBoundingPolygon(FOVh, FOVv, altitude, roll, pitch, heading):
        """Get corners of the polygon captured by the camera on the ground.
        The calculations are performed in the axes origin (0, 0, altitude)
        and the points are not yet translated to camera's X-Y coordinates.
        Parameters:
            FOVh (float): Horizontal field of view in radians
            FOVv (float): Vertical field of view in radians
            altitude (float): Altitude of the camera in meters
            heading (float): Heading of the camera (z axis) in radians
            roll (float): Roll of the camera (x axis) in radians
            pitch (float): Pitch of the camera (y axis) in radians
        Returns:
            vector3d.vector.Vector: Array with 4 points defining a polygon
        """
        # import ipdb; ipdb.set_trace()
        ray11 = CameraCalculator.ray1(FOVh, FOVv)
        ray22 = CameraCalculator.ray2(FOVh, FOVv)
        ray33 = CameraCalculator.ray3(FOVh, FOVv)
        ray44 = CameraCalculator.ray4(FOVh, FOVv)

        rotatedVectors = CameraCalculator.rotateRays(ray11, ray22, ray33, ray44,
                                                      roll, pitch, heading)

        origin = Vector(0, 0, altitude)
        intersections = CameraCalculator.getRayGroundIntersections(rotatedVectors, origin)

        roll2, pitch2, heading2, FOVv2, FOVh2 = CameraCalculator.getFovRPH(intersections, altitude)
        return intersections

    @staticmethod
    def getFovRPH(intersections, altitude):
        # Calculate unit vectors to the ground, assuming camera is at the origin
        rotVecs = [Vector(i.x, i.y, -altitude).normalize() for i in intersections]

        # First rotation aligns the centroid of the polygon with the negative z axis
        centroidVec = (rotVecs[0] + rotVecs[1] + rotVecs[2] + rotVecs[3]).normalize()
        rot1 = rotation_matrix_from_vectors(Vector(z=-1), centroidVec).T

        # Get vectors after first rotation
        rotVecs1 = [Vector(*(rot1 @ np.array([[r.x], [r.y], [r.z]])).flatten()) for r in rotVecs]

        # Second rotation alligns the centroid of the polygon with the negative y axis
        rot2 = rotation_matrix_from_vectors(Vector(y=1), (rotVecs1[0] - rotVecs1[1]).normalize()).T

        # Get final rotation matrix
        R = rot2 @ rot1

        # Calculate roll, pitch and heading
        roll, pitch, heading = roll_pitch_yaw_from_matrix(R.T)

        # Calculate FOVv and FOVh
        rays2 = CameraCalculator.unrotateRays(*rotVecs, roll, pitch, heading)
        rays2norm = [Vector(ray.x / -ray.z, ray.y / -ray.z, -1) for i, ray in enumerate(rays2)]
        FOVv = np.arctan2(rays2norm[0].x, 1) * 2
        FOVh = np.arctan2(rays2norm[0].y, 1) * 2
        return roll, pitch, heading, FOVv, FOVh

    # Ray-vectors defining the camera's field of view. FOVh and FOVv are interchangeable
    # depending on the camera's orientation
    @staticmethod
    def ray1(FOVh, FOVv):
        """
        Parameters:
            FOVh (float): Horizontal field of view in radians
            FOVv (float): Vertical field of view in radians
        Returns:
            vector3d.vector.Vector: normalised vector
        """
        ray = Vector(math.tan(FOVv / 2), math.tan(FOVh / 2), -1)
        return ray.normalize()

    @staticmethod
    def ray2(FOVh, FOVv):
        """
        Parameters:
            FOVh (float): Horizontal field of view in radians
            FOVv (float): Vertical field of view in radians
        Returns:
            vector3d.vector.Vector: normalised vector
        """
        ray = Vector(math.tan(FOVv / 2), -math.tan(FOVh / 2), -1)
        return ray.normalize()

    @staticmethod
    def ray3(FOVh, FOVv):
        """
        Parameters:
            FOVh (float): Horizontal field of view in radians
            FOVv (float): Vertical field of view in radians
        Returns:
            vector3d.vector.Vector: normalised vector
        """
        ray = Vector(-math.tan(FOVv / 2), -math.tan(FOVh / 2), -1)
        return ray.normalize()

    @staticmethod
    def ray4(FOVh, FOVv):
        """
        Parameters:
            FOVh (float): Horizontal field of view in radians
            FOVv (float): Vertical field of view in radians
        Returns:
            vector3d.vector.Vector: normalised vector
        """
        ray = Vector(-math.tan(FOVv / 2), math.tan(FOVh / 2), -1)
        return ray.normalize()

    @staticmethod
    def rotationMatrix(roll, pitch, yaw):
        """Calculate rotation matrix
        Parameters:
            roll float: Roll rotation
            pitch float: Pitch rotation
            yaw float: Yaw rotation
        Returns:
            Returns new rotated ray-vectors
        """
        sinAlpha = math.sin(yaw)
        sinBeta = math.sin(pitch)
        sinGamma = math.sin(roll)
        cosAlpha = math.cos(yaw)
        cosBeta = math.cos(pitch)
        cosGamma = math.cos(roll)
        m00 = cosAlpha * cosBeta
        m01 = cosAlpha * sinBeta * sinGamma - sinAlpha * cosGamma
        m02 = cosAlpha * sinBeta * cosGamma + sinAlpha * sinGamma
        m10 = sinAlpha * cosBeta
        m11 = sinAlpha * sinBeta * sinGamma + cosAlpha * cosGamma
        m12 = sinAlpha * sinBeta * cosGamma - cosAlpha * sinGamma
        m20 = -sinBeta
        m21 = cosBeta * sinGamma
        m22 = cosBeta * cosGamma

        return np.array([[m00, m01, m02],
                         [m10, m11, m12],
                         [m20, m21, m22]])

    @staticmethod
    def rotateRays(ray1, ray2, ray3, ray4, roll, pitch, yaw):
        """Rotates the four ray-vectors around all 3 axes
        Parameters:
            ray1 (vector3d.vector.Vector): First ray-vector
            ray2 (vector3d.vector.Vector): Second ray-vector
            ray3 (vector3d.vector.Vector): Third ray-vector
            ray4 (vector3d.vector.Vector): Fourth ray-vector
            roll float: Roll rotation
            pitch float: Pitch rotation
            yaw float: Yaw rotation
        Returns:
            Returns new rotated ray-vectors
        """
        rotationMatrix = CameraCalculator.rotationMatrix(roll, pitch, yaw)
        rayMatrix = np.array([[ray.x, ray.y, ray.z] for ray in [ray1, ray2, ray3, ray4]]).T
        rotatedRayMatrix = (rotationMatrix @ rayMatrix).T
        rayArray = [Vector(*rayMatrix.flatten()) for rayMatrix in rotatedRayMatrix]
        return rayArray

    @staticmethod
    def unrotateRays(ray1, ray2, ray3, ray4, roll, pitch, yaw):
        """Unrotates the four ray-vectors around all 3 axes
        Parameters:
            ray1 (vector3d.vector.Vector): First ray-vector
            ray2 (vector3d.vector.Vector): Second ray-vector
            ray3 (vector3d.vector.Vector): Third ray-vector
            ray4 (vector3d.vector.Vector): Fourth ray-vector
            roll float: Roll rotation
            pitch float: Pitch rotation
            yaw float: Yaw rotation
        Returns:
            Returns new rotated ray-vectors
        """
        rotationMatrix = CameraCalculator.rotationMatrix(roll, pitch, yaw).T
        rayMatrix = np.array([[ray.x, ray.y, ray.z] for ray in [ray1, ray2, ray3, ray4]]).T
        rotatedRayMatrix = (rotationMatrix @ rayMatrix).T
        rayArray = [Vector(*rayMatrix.flatten()) for rayMatrix in rotatedRayMatrix]
        return rayArray

    @staticmethod
    def getRayGroundIntersections(rays, origin):
        """
        Finds the intersections of the camera's ray-vectors
        and the ground approximated by a horizontal plane
        Parameters:
            rays (vector3d.vector.Vector[]): Array of 4 ray-vectors
            origin (vector3d.vector.Vector): Position of the camera. The computation were developed
                                            assuming the camera was at the axes origin (0, 0, altitude) and the
                                            results translated by the camera's real position afterwards.
        Returns:
            vector3d.vector.Vector
        """
        # Vector3d [] intersections = new Vector3d[rays.length];
        # for (int i = 0; i < rays.length; i ++) {
        #     intersections[i] = CameraCalculator.findRayGroundIntersection(rays[i], origin);
        # }
        # return intersections

        # 1to1 translation without python syntax optimisation
        # intersections = []
        # for i in range(len(rays)):
        #     intersections.append(CameraCalculator.findRayGroundIntersection(rays[i], origin))
        return [CameraCalculator.findRayGroundIntersection(ray, origin) for ray in rays]

    @staticmethod
    def findRayGroundIntersection(ray, origin):
        """
        Finds a ray-vector's intersection with the ground approximated by a planeç
        Parameters:
            ray (vector3d.vector.Vector): Ray-vector
            origin (vector3d.vector.Vector): Camera's position
        Returns:
            vector3d.vector.Vector
        """
        # Parametric form of an equation
        # P = origin + vector * t
        x = Vector(origin.x, ray.x)
        y = Vector(origin.y, ray.y)
        z = Vector(origin.z, ray.z)

        # Equation of the horizontal plane (ground)
        # -z = 0

        # Calculate t by substituting z
        t = - (z.x / z.y)

        # Substitute t in the original parametric equations to get points of intersection
        return Vector(x.x + x.y * t, y.x + y.y * t, z.x + z.y * t)


def get_camera_footprint(camera):
    # altitude = camera.position[2]
    # try:
    #     pan, tilt = camera.pan_tilt
    # except:
    #     pan, tilt = camera.pan, camera.tilt
    #
    # fov_range_pan = (pan - camera.fov_angle[0] / 2, pan, pan + camera.fov_angle[0] / 2)
    # fov_range_tilt = (tilt - camera.fov_angle[1] / 2, tilt, tilt + camera.fov_angle[1] / 2)
    # x_min = altitude * np.tan(fov_range_tilt[0]) + camera.position[0]
    # x_max = altitude * np.tan(fov_range_tilt[2]) + camera.position[0]
    # y_min = altitude * np.tan(fov_range_pan[0]) + camera.position[1]
    # y_max = altitude * np.tan(fov_range_pan[2]) + camera.position[1]

    # Once the camera is rotated, the z axis becomes the x axis, and the x axis becomes the z axis
    # TODO: More testing is needed to make sure this is correct
    roll, pitch, heading = (camera.orientation[2],
                            camera.orientation[1] + np.pi / 2,
                            camera.orientation[0])

    xmin, xmax, ymin, ymax = get_camera_footprint_low(camera.position, roll, pitch, heading,
                                                      camera.fov_angle)
    return xmin, xmax, ymin, ymax


def get_camera_footprint_low(position, roll, pitch, heading, fov_angle):
    bpol = CameraCalculator.getBoundingPolygon(fov_angle[0],
                                               fov_angle[1],
                                               position[2],
                                               roll,  # Tested, works
                                               -pitch,  # Tested, works
                                               -heading)
    xvals = np.sort(np.unique(np.round([v.x + position[0] for v in bpol], 2)))
    yvals = np.sort(np.unique(np.round([v.y + position[1] for v in bpol], 2)))

    xmin = xvals[0]
    xmax = xvals[-1]
    ymin = yvals[0]
    ymax = yvals[-1]

    return xmin, xmax, ymin, ymax


def get_roll_pitch_yaw_fov(x_min, x_max, y_min, y_max, altitude):
    intersections = [Vector(x_max, y_max), Vector(x_max, y_min), Vector(x_min, y_min),
                     Vector(x_min, y_max)]

    roll, pitch, heading, fov_tilt, fov_pan = CameraCalculator.getFovRPH(intersections, altitude)
    # Pitch needs to be inverted to get the tilt angle
    pitch = -pitch

    return Angle(roll), Angle(pitch), Angle(heading), Angle(fov_tilt), Angle(fov_pan)


def lla_to_pan_tilt_fov(pos, x_min, x_max, y_min, y_max):
    """Converts lat, lon, alt to az, el"""

    # We assume that the camera is looking at the ground, with heading pointing east
    # Hence, panning is along the latitude axis, and tilting is along the longitude axis

    alt = pos[2]
    phi1 = np.arctan2(x_min - pos[0], alt)
    phi2 = np.arctan2(x_max - pos[0], alt)

    theta1 = np.arctan2(y_min - pos[1], alt)
    theta2 = np.arctan2(y_max - pos[1], alt)

    fov_angle = (theta2 - theta1, phi2 - phi1)
    pan, tilt = (fov_angle[0]) / 2 + theta1, (fov_angle[1]) / 2 + phi1

    return pan, tilt, fov_angle


def get_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    if isinstance(vec1, Vector):
        vec1 = np.array([vec1.x, vec1.y, vec1.z])
    if isinstance(vec2, Vector):
        vec2 = np.array([vec2.x, vec2.y, vec2.z])
    if np.allclose(vec1, vec2):
        return np.eye(3)
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def roll_pitch_yaw_from_matrix(matrix):
    """ Returns the Euler angles from a rotation matrix
    :param matrix: A transform matrix (3x3)
    :return: Euler angles in the form of a tuple (roll, pitch, yaw)
    """
    roll = np.arctan2(matrix[2, 1], matrix[2, 2])
    pitch = np.arctan2(-matrix[2, 0], np.sqrt(matrix[2, 1] ** 2 + matrix[2, 2] ** 2))
    yaw = np.arctan2(matrix[1, 0], matrix[0, 0])
    return roll, pitch, yaw


def rigid_transform_3D(A, B):
    assert A.shape == B.shape
    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")
    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")
    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)
    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)
    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B
    H = Am @ np.transpose(Bm)
    # sanity check
    # if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))
    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    t = -R @ centroid_A + centroid_B
    return R, t


def calculate_num_targets_dist(tracks: Set[Track], geom: BaseGeometry,
                               phd_state: ParticleState = None, target_types: List[str] = None):
    num_samples = 100
    valid_tracks = [track for track in tracks
                    if not (target_types)
                    or (target_types and any(item in track.metadata['target_type_confidences']
                                             for item in target_types))]
    mu_overall = 0
    var_overall = np.inf if len(valid_tracks) == 0 else 0
    path_p = Path(geom.boundary.coords)

    # Calculate PHD density inside polygon
    if phd_state is not None:
        points = phd_state.state_vector[[0, 2], :].T
        inside_points = path_p.contains_points(points)
        if np.sum(inside_points) > 0:
            # The mean of the PHD density inside the polygon is the sum of the weights of the
            # particles inside the polygon
            mu_overall = np.exp(logsumexp(np.log(phd_state.weight[inside_points].astype(float))))
            # The variance of a Poisson distribution is equal to the mean
            var_overall = mu_overall

    # Calculate number of tracks inside polygon
    for track in valid_tracks:
        # Sample points from the track state
        points = multivariate_normal.rvs(mean=track.state_vector[[0, 2]].ravel(),
                                         cov=track.covar[[0, 2], :][:, [0, 2]],
                                         size=num_samples)
        # Check which points are inside the polygon
        inside_points = path_p.contains_points(points)
        # Probability of existence inside the polygon is the fraction of points inside the polygon
        # times the probability of existence
        p_success = float(track.exist_prob) * (np.sum(inside_points) / num_samples)
        # Mean of a Bernoulli distribution is equal to the probability of success
        mu_overall += p_success
        # Variance of a Bernoulli distribution is equal to the probability of success,
        # times the probability of failure
        var_overall += p_success * (1 - p_success)

    return mu_overall, var_overall
