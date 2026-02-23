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
    # There is (z - 0.99)^2, I think that zero gets included twice:
    zeros = [0, 0.8, -0.99, -0.99]
    b = np.poly(zeros)
    a = np.poly(poles)
    return (b, a)


def inverse():
    """
    This is the function to undo the aberration
    """
    b, a = filter()
    return (a, b)


def fix(data: np.ndarray) -> np.ndarray:
    """
    Apply the inverse filter from `inverse()` to every row and column in the input data
    """
    b, a = inverse()
    # TODO: Why must I invert a and b for this shit work??? There is a fuck
    # somewhere, right?
    return img.apply(a, b, data)


def main():

    b, a = filter()
    zplane(b, a, "aberration-zplane-orig.pdf")
    plt.close()
    b, a = inverse()
    zplane(b, a, "aberration-zplane-inv.pdf")
    plt.close()

    orig = img.aberrated()
    img.save("aberration-before.png", orig)
    fixed = fix(orig)
    img.save("aberration-after.png", fixed)


if __name__ == "__main__":
    main()
