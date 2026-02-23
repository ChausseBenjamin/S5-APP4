import numpy as np

def fix(data):
    """
    returns an image rotated by 90 degrees
    TODO: Implement both methods and use one of them here
    """
    return data

def get_transformation_array(angle_rad: float):
    """
        Builds a pixel coordinate transformation table
        so 2x2
    """
    sin = np.sin(angle_rad)
    cos = np.cos(angle_rad)

    return np.array([
    [cos, sin],
    [sin, cos]
    ])