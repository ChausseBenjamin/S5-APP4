import numpy as np
import img


def fix(data):
    """
    returns an image rotated by 90 degrees
    TODO: Implement both methods and use one of them here
    """
    data = rotate_picture(data, 90)
    # for i in range(1,700):
    #     data = rotate_picture(data, 45.1)

    return data


def get_transformation_array(angle: float):
    """
    Builds a pixel coordinate transformation table
    so 2x2
    """
    radians = np.deg2rad(angle)
    sin = np.sin(radians)
    cos = np.cos(radians)

    sin = np.round(sin, decimals=10)
    cos = np.round(cos, decimals=10)

    return np.array([[cos, -sin], [sin, cos]])


def rotate_picture(image: np.ndarray, angle: float):

    R = get_transformation_array(angle)

    h, w = image.shape
    rotated = np.zeros((h, w))

    cx = (w - 1) / 2
    cy = (h - 1) / 2

    # Output grid (destination pixels)
    x = np.arange(w)
    y = np.arange(h)
    xx, yy = np.meshgrid(x, y)

    # Shift origin to center
    x_shifted = xx - cx
    y_shifted = yy - cy

    coords = np.vstack((x_shifted.ravel(), y_shifted.ravel()))

    # Map BACK into source image
    source_coords = R.T @ coords

    ip = (source_coords[0] + cx).round().astype(int)
    jp = (source_coords[1] + cy).round().astype(int)

    ip = ip.reshape(h, w)
    jp = jp.reshape(h, w)

    mask = (0 <= ip) & (ip < w) & (0 <= jp) & (jp < h)

    rotated[mask] = image[jp[mask], ip[mask]]

    return rotated


def main():
    image = img.rotated()
    img.save("rotated-orig.png", image)
    rotated = rotate_picture(image, 45)
    img.save("rotated-example.png", rotated)

    image = img.philippe()
    rotated = rotate_picture(image, angle=45/2)
    img.save("rotated-22.png", rotated)
    rotated = rotate_picture(image, angle=45)
    img.save("rotated-45.png", rotated)
    rotated = rotate_picture(image, angle=45 + 45/2)
    img.save("rotated-67.png", rotated)

if __name__ == "__main__":
    main()
