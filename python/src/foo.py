"""
Test script to confirm img.py can be used both as a standalone script or be
imported as a lib
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
from src.img import original, _output


def colvec(n):
    data = original()
    return data[:, n]


def col1plt():
    col = colvec(0)
    plt.figure()
    plt.plot(col)
    plt.title("First column")
    plt.savefig(f"{_output}/test_foo_col1.png")


def main():
    col1plt()


if __name__ == "__main__":
    main()
