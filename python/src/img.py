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
import scipy.signal as sig

# Directory constants
_input = "input"
_output = "output"


def load_png(path: str) -> np.ndarray:
    plt.gray()
    data = mpimg.imread(path)
    if data.ndim == 3:
        return np.mean(data, -1)
    return data


def original() -> np.ndarray:
    """
    Load the original image as npy data
    """
    return load_png(f"{_input}/goldhill.png")


def aberrated() -> np.ndarray:
    return np.load(f"{_input}/goldhill_aberrations.npy")


def noisy() -> np.ndarray:
    return np.load(f"{_input}/goldhill_bruit.npy")


def rotated() -> np.ndarray:
    return load_png(f"{_input}/goldhill_rotate.png")


def complete() -> np.ndarray:
    return np.load(f"{_input}/image_complete.npy")

def philippe() -> np.ndarray:
    return load_png(f"{_input}/IMG-Philippe-Gournay-20260202111650.png")

def apply(b: np.ndarray, a: np.ndarray, data: np.ndarray):
    return sig.lfilter(b, a, data.copy())


def save(filename: str, data: np.ndarray) -> None:
    """
    Save npy data as a png into the default output directory
    """
    plt.imsave(f"{_output}/{filename}", data, cmap="gray")


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
