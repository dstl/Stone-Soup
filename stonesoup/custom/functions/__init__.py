import numpy as np


def get_camera_footprint(camera):
    pan, tilt = camera.pan_tilt
    altitude = camera.position[2]

    fov_range_pan = (pan-camera.fov_angle[0]/2, pan, pan+camera.fov_angle[0]/2)
    fov_range_tilt = (tilt-camera.fov_angle[1]/2, tilt, tilt+camera.fov_angle[1]/2)
    x_min = altitude * np.tan(fov_range_tilt[0]) + camera.position[0]
    x_max = altitude * np.tan(fov_range_tilt[2]) + camera.position[0]
    y_min = altitude * np.tan(fov_range_pan[0]) + camera.position[1]
    y_max = altitude * np.tan(fov_range_pan[2]) + camera.position[1]
    return x_min, x_max, y_min, y_max


def get_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]