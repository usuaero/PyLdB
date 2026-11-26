"""Reconstruct one-third-octave sound exposure with normalized Morlet gains."""

import numpy as np


def _normalized_morlet_gains(frequencies_hz, lower_hz, upper_hz):
    """Return amplitude gains indexed by wavelet scale and frequency bin."""
    log_edges = np.linspace(np.log(lower_hz), np.log(upper_hz), 7)
    centers_hz = np.exp((log_edges[:-1] + log_edges[1:]) / 2)
    morlet_omega0 = 6.0  # Dimensionless Morlet center frequency.
    omega = morlet_omega0 * frequencies_hz[np.newaxis, :] / centers_hz[:, np.newaxis]
    # Factor out the largest zero-mean Morlet gain before normalizing each column.
    log_gain = -0.5 * (omega - morlet_omega0) ** 2 + np.log(-np.expm1(-morlet_omega0 * omega))
    gains = np.exp(log_gain - log_gain.max(axis=0))
    return gains / np.linalg.norm(gains, axis=0)


def _band_exposure_rate(spectrum, frequencies_hz, sample_count, band, bins, fraction):
    """Return the exposure rate of a band from an exposure-scaled one-sided FFT."""
    lower_hz, upper_hz = band
    allocated = spectrum[bins] * np.sqrt(fraction)
    active = np.flatnonzero(allocated)
    if not active.size:
        return np.zeros(sample_count)

    band_spectrum = allocated[active]
    indices = bins.start + active
    timeline = np.zeros(sample_count)
    if indices[0] == 0:
        # The zero-mean wavelets have no DC response. Preserve the allocated
        # DC exposure as a constant rate over the record.
        timeline.fill((abs(band_spectrum[0]) / sample_count) ** 2)
        indices = indices[1:]
        band_spectrum = band_spectrum[1:]

    if indices.size:
        gains = _normalized_morlet_gains(frequencies_hz[indices], lower_hz, upper_hz)
        wavelet_spectra = np.zeros((gains.shape[0], sample_count), dtype=complex)
        wavelet_spectra[:, indices] = gains * band_spectrum
        responses = np.fft.ifft(wavelet_spectra, axis=-1)
        timeline += np.sum(np.abs(responses) ** 2, axis=0)
    return timeline
