import numpy as np
from numpy import pi


def polar(theta, r=1):
    """
    returns a valid python complex number z = (a + bj)
    given that complex number in it's polar form: z = r * e**(j * theta)
    """
    return r * np.exp(1j * theta)


def main():
    polarTests = [
        # theta (radians), radius, expected
        (0, 1, 1 + 0j),
        (pi / 2, 2, 0 + 2j),
        (pi, 3, -3 + 0j),
        (3 * pi / 2, 4, 0 - 4j),
        (2 * pi, 5, 5 + 0j),
    ]
    for case in polarTests:
        theta, r, expected = case
        z = polar(theta, r)
        if np.isclose(z, expected):
            print(f"VALID: {r}e^(j{theta}) = {z}")
        else:
            print(f"ERROR! expected: {r}e^(j{theta}) = {expected}, got {z}")


if __name__ == "__main__":
    main()
