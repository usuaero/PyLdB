"""Verify full-record Mark VII PLdB using Morlet filters with normalized gains."""

from dataclasses import dataclass

import numpy as np

from compare_mark_vii import PA_PER_PSF, waveforms
from pyldb import PyLdB


@dataclass(frozen=True)
class WaveletComparison:
    fft_exposure_pa2_s: np.ndarray
    wavelet_exposure_pa2_s: np.ndarray
    fft_pldb: float
    wavelet_pldb: float


def compare_waveform(time_ms, pressure_pa, taper, padding):
    """Compare the two public PyLdB methods for the same pressure signature."""
    calculator = PyLdB(time=time_ms, pressure=pressure_pa / PA_PER_PSF)
    fft_pldb = calculator.perceived_loudness(
        len_window=taper, pad_front=padding, pad_rear=padding,
    )
    fft_exposure = calculator.band_energy * (calculator.ref_time_s * PA_PER_PSF**2)
    wavelet_pldb = calculator.perceived_loudness(
        len_window=taper, pad_front=padding, pad_rear=padding, method="wavelet",
    )
    return WaveletComparison(
        fft_exposure_pa2_s=fft_exposure,
        wavelet_exposure_pa2_s=calculator.wavelet_exposure.band_exposure_pa2_s,
        fft_pldb=fft_pldb,
        wavelet_pldb=wavelet_pldb,
    )


if __name__ == "__main__":
    print("| Waveform | FFT PLdB | Wavelet PLdB | PLdB difference | Largest band exposure difference (Pa^2 s) |")
    print("|---|---:|---:|---:|---:|")
    for name, *inputs in waveforms():
        result = compare_waveform(*inputs)
        pldb_difference = result.wavelet_pldb - result.fft_pldb
        band_difference = np.max(np.abs(result.wavelet_exposure_pa2_s - result.fft_exposure_pa2_s))
        print(
            f"| {name} | {result.fft_pldb:.8f} | {result.wavelet_pldb:.8f} "
            f"| {pldb_difference:+.3e} | {band_difference:.3e} |"
        )
