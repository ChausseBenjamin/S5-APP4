#
# Copyright (c) 2011 Christopher Felton
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

# The following is derived from the slides presented by
# Alexander Kain for CS506/606 "Special Topics: Speech Signal Processing"
# CSLU / OHSU, Spring Term 2011.

import sys
import os

import matplotlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from img import _output
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches


def zplane(b, a, filename: str | None = None):
    """Plot the complex z-plane given a transfer function."""

    # get a figure/plot
    ax = plt.subplot(111)

    # create the unit circle
    uc = patches.Circle((0, 0), radius=1, fill=False, color="black", ls="dashed")
    ax.add_patch(uc)

    # The coefficients are less than 1, normalize the coeficients
    if np.max(b) > 1:
        kn = np.max(b)
        b = b / float(kn)
    else:
        kn = 1

    if np.max(a) > 1:
        kd = np.max(a)
        a = a / float(kd)
    else:
        kd = 1

    # Get the poles and zeros
    p = np.roots(a)
    z = np.roots(b)
    k = kn / float(kd)

    # Plot the zeros and set marker properties
    t1 = plt.plot(z.real, z.imag, "go", ms=10)
    plt.setp(
        t1,
        markersize=10.0,
        markeredgewidth=1.0,
        markeredgecolor="k",
        markerfacecolor="g",
    )

    # Plot the poles and set marker properties
    t2 = plt.plot(p.real, p.imag, "rx", ms=10)
    plt.setp(
        t2,
        markersize=12.0,
        markeredgewidth=3.0,
        markeredgecolor="r",
        markerfacecolor="r",
    )

    ax.spines["left"].set_position("center")
    ax.spines["bottom"].set_position("center")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # set the ticks
    r = 1.5
    plt.axis("scaled")
    plt.axis([-r, r, -r, r])
    ticks = [-1, -0.5, 0.5, 1]
    plt.xticks(ticks)
    plt.yticks(ticks)

    if filename is None:
        plt.show()
    else:
        plt.savefig(f"{_output}/{filename}", bbox_inches="tight")

    return z, p, k


def main():
    matplotlib.use("qt5agg")
    zeros = [0, 0.2 + 0.8j, 0.2 - 0.8j]
    poles = [0.9, -0.8 + 0.2j, -0.8 - 0.2j]
    b = np.poly(zeros)  # numerator
    a = np.poly(poles)  # denominator
    zplane(b, a)


if __name__ == "__main__":
    main()
