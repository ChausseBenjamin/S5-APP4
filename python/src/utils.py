import numpy as np
from numpy import pi


def polar(theta: float, r: int | float = 1.0) -> complex:
    """
    returns a valid python complex number z = (a + bj)
    given that complex number in it's polar form: z = r * e**(j * theta)
    """
    return r * np.exp(1j * theta)

def is_orthogonal(matrix: np.ndarray, tol: float = 1e-8) -> bool:
    """
    Checks if a matrix is orthogonal:
    Q^T Q = I
    
    Parameters:
        matrix : np.ndarray
        tol : tolerance for floating point comparison
        
    Returns:
        bool
    """
    if matrix.shape[0] != matrix.shape[1]:
        return False  # Must be square
    
    identity = np.eye(matrix.shape[0])
    product = matrix.T @ matrix
    
    return np.allclose(product, identity, atol=tol)

def dB2Amp(val: int | float | np.ndarray) -> float | np.ndarray:
    return np.power(10.0, np.asarray(val, dtype=float) / 20.0)


def amp2dB(val: int | float | np.ndarray) -> float | np.ndarray:
    val_arr = np.asarray(val, dtype=float)
    return 20.0 * np.log10(val_arr)


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

    # dB <-> Amplitude tests
    dbTests = [-60, -20, 0, 6]
    for db in dbTests:
        amp = dB2Amp(db)
        db_back = amp2dB(amp)
        if np.isclose(db, db_back):
            print(f"VALID: {db} dB -> {amp} -> {db_back} dB")
        else:
            print(f"ERROR! {db} dB roundtrip failed (got {db_back})")

    # vector test
    db_vec = np.array([-60, -20, 0])
    amp_vec = dB2Amp(db_vec)
    db_vec_back = amp2dB(amp_vec)
    if np.allclose(db_vec, db_vec_back):
        print("VALID: vector dB roundtrip")
    else:
        print("ERROR! vector dB roundtrip failed")


if __name__ == "__main__":
    main()
