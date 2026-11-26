"""Verify full-record Mark VII PLdB using Morlet filters with normalized gains."""

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from compare_mark_vii import PA_PER_PSF, waveforms
from mark_vii_reference import BAND_EDGES_HZ
from pyldb import PyLdB


def _normalized_morlet_gains(frequencies_hz, lower_hz, upper_hz):
    """Return amplitude gains indexed by wavelet scale and frequency bin."""
    log_edges = np.linspace(np.log(lower_hz), np.log(upper_hz), 7)
    centers_hz = np.exp((log_edges[:-1] + log_edges[1:]) / 2)
    morlet_omega0 = 6.0  # Dimensionless Morlet center frequency.
    omega = morlet_omega0 * frequencies_hz[np.newaxis, :] / centers_hz[:, np.newaxis]
    # Zero-mean Morlet response at positive frequencies. Work in log space
    # and factor out the largest gain before normalizing each column.
    log_gain = -0.5 * (omega - morlet_omega0) ** 2 + np.log(-np.expm1(-morlet_omega0 * omega))
    gains = np.exp(log_gain - log_gain.max(axis=0))
    return gains / np.linalg.norm(gains, axis=0)


def band_exposure_timelines(pressure_pa, spacing_s) -> Iterator[np.ndarray]:
    """Yield one Pa² timeline per band on the processed record's circular DFT grid."""
    sample_count = len(pressure_pa)
    frequencies = np.fft.rfftfreq(sample_count, spacing_s)
    spectrum = np.fft.rfft(pressure_pa)
    bin_width = frequencies[1]
    bin_left = np.maximum(0.0, frequencies - bin_width / 2)
    bin_right = np.minimum(0.5 / spacing_s, frequencies + bin_width / 2)

    # The complex inverse FFT uses the nonnegative-frequency bins here. Scale
    # paired bins by sqrt(2) to retain exposure. DC and even Nyquist are unpaired.
    paired = slice(1, -1 if sample_count % 2 == 0 else None)
    spectrum[paired] *= np.sqrt(2.0)

    for lower, upper in zip(BAND_EDGES_HZ[:-1], BAND_EDGES_HZ[1:]):
        bins = np.flatnonzero((bin_right > lower) & (bin_left < upper) & (spectrum != 0))
        overlap = np.minimum(bin_right[bins], upper) - np.maximum(bin_left[bins], lower)
        fraction = overlap / (bin_right[bins] - bin_left[bins])
        band_spectrum = spectrum[bins] * np.sqrt(fraction)

        timeline = np.zeros(sample_count)
        if bins.size and bins[0] == 0:
            # An admissible wavelet has zero DC gain. Carry the assigned DC
            # contribution as a constant rate over the DFT record.
            timeline.fill((abs(band_spectrum[0]) / sample_count) ** 2)
            bins = bins[1:]
            band_spectrum = band_spectrum[1:]

        if bins.size:
            gains = _normalized_morlet_gains(frequencies[bins], lower, upper)
            wavelet_spectra = np.zeros((gains.shape[0], sample_count), dtype=complex)
            wavelet_spectra[:, bins] = gains * band_spectrum
            responses = np.fft.ifft(wavelet_spectra, axis=-1)
            timeline += np.sum(np.abs(responses) ** 2, axis=0)
        yield timeline


@dataclass(frozen=True)
class WaveletComparison:
    fft_exposure_pa2_s: np.ndarray
    wavelet_exposure_pa2_s: np.ndarray
    fft_pldb: float
    wavelet_pldb: float


def compare_waveform(time_ms, pressure_pa, taper, padding):
    """Compare both calculations using the same PyLdB preprocessing and Mark VII stages."""
    calculator = PyLdB(time=time_ms, pressure=pressure_pa / PA_PER_PSF)
    fft_pldb = calculator.perceived_loudness(
        len_window=taper, pad_front=padding, pad_rear=padding,
    )
    spacing_s = calculator.dt_s
    processed_pa = calculator.sig_pressure_psf * PA_PER_PSF
    wavelet_exposure = np.array([
        timeline.sum() * spacing_s for timeline in band_exposure_timelines(processed_pa, spacing_s)
    ])
    # Match PyLdB's pressure reference, including its conversion precision.
    reference_pressure_pa = calculator.ref_pressure_psf * PA_PER_PSF
    with np.errstate(divide="ignore"):
        wavelet_spl = 10 * np.log10(
            wavelet_exposure / (calculator.ref_time_s * reference_pressure_pa**2)
        ) - 3
    equivalent = calculator._equivalent_loudness(wavelet_spl)
    total_sones, _ = calculator._calc_total_loudness(equivalent)
    return WaveletComparison(
        fft_exposure_pa2_s=calculator.band_energy * (calculator.ref_time_s * PA_PER_PSF**2),
        wavelet_exposure_pa2_s=wavelet_exposure,
        fft_pldb=fft_pldb,
        wavelet_pldb=calculator._sones_to_pldb(total_sones),
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
