import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from numpy import pi
from scipy import signal
from utils import polar
from zplane import zplane
import img
import matplotlib.pyplot as plt
import numpy as np


def filter():
    """
    This is the original aberration function, not its inverse.
    """
    poles = [
        polar(pi / 2, 0.9),
        polar(-pi / 2, 0.9),
        polar(pi / 8, 0.95),
        polar(-pi / 8, 0.95),
    ]
    zeros = [0, 0.8, -0.99]
    b = np.poly(zeros)
    a = np.poly(poles)
    return (a, b)


def inverse():
    """
    This is the function to undo the aberration
    """
    a, b = filter()
    return (b, a)


def fix(data: np.ndarray) -> np.ndarray:
    """
    Apply the inverse filter from `inverse()` to every row and column in the input data
    """
    a, b = inverse()

    result = data.copy()
    for col in range(result.shape[1]):
        result[:, col] = signal.lfilter(a, b, result[:, col])
    for row in range(result.shape[0]):
        result[row, :] = signal.lfilter(a, b, result[row, :])
    return result


def main():

    a, b = filter()
    zplane(a, b, "aberration-zplane-orig.pdf")
    plt.close()
    a, b = inverse()
    zplane(a, b, "aberration-zplane-inv.pdf")
    plt.close()

    orig = img.aberrated()
    img.save("aberration-before.png", orig)
    fixed = fix(orig)
    img.save("aberration-after.png", fixed)


if __name__ == "__main__":
    main()
