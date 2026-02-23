import sys
import os

from zplane import zplane

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import amp2dB
from img import _output

from scipy.signal import sosfreqz
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig

# Criterias
# - Assume the sampling rate is 1600 Hz
# - Hover between -0.2 and 0.2 dB from 0 to 500 Hz (main)
# - Max -60dB for everything over 750 Hz (damp)
_rate = 1600  # Hz
_cutoff = 500
_damp_target = 750
_damp_amp = -60
_fluctuation = 0.2
_nyquist = 1600 / 2


def butter(n: int = 1, output: str = "sos"):
    return sig.butter(n, _cutoff, btype="low", fs=_rate, output=output)


def cheby1(n: int = 1, output: str = "sos"):
    return sig.cheby1(n, _fluctuation, _cutoff, btype="low", fs=_rate, output=output)


def cheby2(n: int = 1, output: str = "sos"):
    return sig.cheby2(n, abs(_damp_amp), _cutoff, btype="low", fs=_rate, output=output)


def ellip(n: int = 1, output: str = "sos"):
    return sig.ellip(
        n, _fluctuation, abs(_damp_amp), _cutoff, btype="low", fs=_rate, output=output
    )


def overlay(ax):
    """
    Draws the passband and stopband criteria on the given axes.
    """
    # Cutoff line
    ax.axvline(_cutoff, color="seagreen", linestyle="--", label=f"$F_c = {_cutoff}$ Hz")

    # Allowed fluctuation region (passband)
    ax.fill_between(
        [0, _cutoff],
        -_fluctuation,
        _fluctuation,
        color="teal",
        alpha=0.2,
        label=f"Fluctuation ±{_fluctuation} dB (permise)",
    )

    # Forbidden stopband region
    ylim_top = ax.get_ylim()[1]
    ax.fill_between(
        [_damp_target, _nyquist],
        _damp_amp,
        ylim_top,
        color="red",
        alpha=0.2,
        label=f"Amplitude > {_damp_amp} dB (interdite)",
    )


def freqplot(sos):
    return sosfreqz(sos, fs=_rate)


def passesCriterias(w, h_db):
    """
    Check if the filter response meets the passband and stopband criteria.
    Returns:
        passed (bool): True if all criteria are met
        coast_min (float): min amplitude in passband
        coast_max (float): max amplitude in passband
        damped_max (float): max amplitude in stopband
    """
    criterias = []

    # Passband: 0 Hz -> _cutoff
    mask_main = (w >= 0) & (w <= _cutoff)
    coast_max = np.max(h_db[mask_main])
    coast_min = np.min(h_db[mask_main])
    criterias.append(coast_max <= _fluctuation)
    criterias.append(coast_min >= -_fluctuation)

    # Stopband: _damp_target -> _nyquist
    mask_stop = w >= _damp_target
    damped_max = np.max(h_db[mask_stop])
    criterias.append(damped_max <= _damp_amp)

    full = all(criterias)
    return full, coast_min, coast_max, damped_max


def multiNPlot(filter, ax, name: str = "Filtre", max: int = 12):
    """
    Plot increasing orders of a filter on the given axes until one meets the criteria.
    """
    for n in range(1, max):
        sos = filter(n=n)
        w, h = freqplot(sos)
        h_db = amp2dB(np.abs(h))
        ax.semilogx(w, h_db, label=f"{name} d'ordre {n}")

        full_pass, coast_min, coast_max, damped_max = passesCriterias(w, h_db)
        if full_pass:
            print(f"Le filtre '{name}' d'ordre {n} rencontre tout les critères")
            print(f"\t Min/Max de 0 à {_cutoff} Hz: {coast_min}, {coast_max} ")
            print(f"\t Max de {_damp_target} à {_nyquist} Hz: {damped_max}")
            return  # stop at first passing order
    print(
        f"Aucun filtre '{name}' ne rencontre tout les criteres jusqu'a un ordre {max}"
    )


def main():
    types = {
        "Butterworth": butter,
        "Chebyshev I": cheby1,
        "Chebyshev II": cheby2,
        "Elliptic": ellip,
    }

    for name, filter_func in types.items():
        fig, ax = plt.subplots(figsize=(11.5, 4.5))

        multiNPlot(filter_func, ax, name)

        overlay(ax)

        ax.set_ylim(-70, 3)
        ax.set_xlim(0, _nyquist)
        ax.set_xlabel("Fréquence [Hz]")
        ax.set_ylabel("Amplitude [dB]")
        ax.grid(True, which="both", ls="--", alpha=0.3)
        ax.legend()

        fig.savefig(
            f"{_output}/denoise-{name.replace(' ', '-').lower()}-all.pdf",
            bbox_inches="tight",
        )
        plt.close(fig)

    # Chosen filter alone (Order 3 Elliptical)
    fig, ax = plt.subplots(figsize=(11.5, 4.5))
    sos = ellip(3)
    w, h = freqplot(np.abs(sos))
    h_db = amp2dB(np.abs(h))
    ax.semilogx(w, h_db, label="Filtre Elliptique d'ordre 3")
    overlay(ax)
    ax.set_ylim(-70, 3)
    ax.set_xlim(0, _nyquist)
    ax.set_xlabel("Fréquence [Hz]")
    ax.set_ylabel("Amplitude [dB]")
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.legend()
    fig.savefig(
        f"{_output}/denoise-chosen-freq.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

    # Poles and zeroes of chosen filter
    b, a = ellip(3, "ba")
    zplane(b, a, "denoise-chosen-zplane.pdf")


if __name__ == "__main__":
    main()
