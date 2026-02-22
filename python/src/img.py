"""
Allows loading stuff from the inputs directory

Functions bear the name of the image/data being loaded
so that calling them externally looks like:
    - `img.original()`
    - `img.aberrated()`
    - `img.noisy()`
    - `img.rotated()`
    - `img.complete()`
    - `img.save("processed_v1.png", my_data)`
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Directory constants
_input = "input"
_output = "output"


def load_png(path):
    plt.gray()
    data = mpimg.imread(path)
    if data.ndim == 3:
        return np.mean(data, -1)
    return data


def original():
    """
    Load the original image as npy data
    """
    return load_png(f"{_input}/goldhill.png")


def aberrated():
    return np.load(f"{_input}/goldhill_aberrations.npy")


def noisy():
    return np.load(f"{_input}/goldhill_bruit.npy")


def rotated():
    return load_png(f"{_input}/goldhill_rotate.png")


def complete():
    return np.load(f"{_input}/image_complete.npy")


def save(filename, data):
    """
    Save npy data as a png into the default output directory
    """
    plt.imsave(f"{_output}/{filename}", data)


def main():
    # Try to load and save each input
    cases = [
        ("orig", original()),
        ("aberrated", aberrated()),
        ("noisy", noisy()),
        ("rotated", rotated()),
        ("complete", complete()),
    ]

    for case in cases:
        ext = case[0]
        data = case[1]
        save(f"test_loadsave_{ext}.png", data)


if __name__ == "__main__":
    main()
