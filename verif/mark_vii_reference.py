"""Independent Mark VII reference for the verification suite, with SI inputs."""

from dataclasses import dataclass

import numpy as np


# Base-10 third octaves, nominal centers 1 Hz through 12.5 kHz.
BAND_NUMBERS = np.arange(42)
BAND_CENTERS_HZ = 10.0 ** (BAND_NUMBERS / 10)
BAND_EDGES_HZ = 10.0 ** ((np.arange(43) - 0.5) / 10)
REFERENCE_PRESSURE_PA = 20e-6
REFERENCE_TIME_S = 0.070
QUIET_PRESSURE_RATIO = 10.0 ** (-0.3)
ONE_SONE_PRESSURE_RATIO = 10.0 ** 3.2

# Stevens (1972), Appendix B, as reprinted in the General Radio handbook,
# Table 3-4. The first two pairs are the additions in Klos (2025), Table 1.
SUMMATION_SONES = (
    0, .113, .181, .196, .212, .230, .248, .269, .290, .314, .339, .367,
    .396, .428, .463, .500, .540, .583, .630, .680, .735, .794, .857, .926,
    1, 1.08, 1.17, 1.26, 1.36, 1.47, 1.59, 1.72, 1.85, 2, 2.16, 2.33,
    2.52, 2.72, 2.94, 3.18, 3.43, 3.70, 4, 4.32, 4.67, 5.04, 5.44,
    5.88, 6.35, 6.86, 7.41, 8, 8.64, 9.33, 10.1, 10.9, 11.8, 12.7, 13.7,
    14.8, 16, 17.3, 18.7, 20.2, 21.8, 23.5, 25.4, 27.4, 29.6, 32, 34.6,
    37.3, 40.3, 43.5, 47, 50.8, 54.9, 59.3, 64, 69.1, 74.7, 80.6, 87.1,
    94.1, 102, 110, 119, 128, 138, 149, 161, 174, 188, 203, 219, 237, 256,
)
SUMMATION_FACTORS = (
    0, 0, .100, .122, .140, .158, .174, .187, .200, .212, .222, .232,
    .241, .250, .259, .267, .274, .281, .287, .293, .298, .303, .308, .312,
    .316, .319, .320, .322, .322, .320, .319, .317, .314, .311, .308, .304,
    .300, .296, .292, .288, .284, .279, .275, .270, .266, .262, .258,
    .253, .248, .244, .240, .235, .230, .226, .222, .217, .212, .208, .204,
    .200, .197, .195, .194, .193, .192, .191, .190, .190, .190, .190, .190,
    .190, .191, .191, .192, .193, .194, .195, .197, .199, .201, .203, .205,
    .208, .210, .212, .215, .217, .219, .221, .223, .224, .225, .226,
    .227, .227, .227,
)


@dataclass(frozen=True)
class LoudnessSpectrum:
    equivalent_level_db: np.ndarray
    sones: np.ndarray
    total_sones: float
    pldb: float


@dataclass(frozen=True)
class PerceivedLevel:
    band_energy_pa2_s: np.ndarray
    band_spl_db: np.ndarray
    loudness: LoudnessSpectrum


def fft_bin_energy(pressure_pa, sample_spacing_s):
    """Return one-sided bin centers and their integrated energy in Pa² s."""
    sample_count = len(pressure_pa)
    frequencies = np.fft.rfftfreq(sample_count, sample_spacing_s)
    transform = np.fft.rfft(pressure_pa)
    energy = np.abs(transform) ** 2 * (sample_spacing_s / sample_count)
    # Only bins with a distinct negative-frequency partner are doubled.
    positive_pairs = slice(1, -1 if sample_count % 2 == 0 else None)
    energy[positive_pairs] *= 2
    return frequencies, energy


def third_octave_energy(pressure_pa, sample_spacing_s):
    """Allocate each FFT bin's energy by overlap (Klos 2025, Appendix 1)."""
    frequencies, energy = fft_bin_energy(pressure_pa, sample_spacing_s)
    half_bin = frequencies[1] / 2
    left = np.maximum(0, frequencies - half_bin)
    right = np.minimum(0.5 / sample_spacing_s, frequencies + half_bin)
    band_energy = np.zeros(len(BAND_NUMBERS))
    for index, (lower, upper) in enumerate(zip(BAND_EDGES_HZ, BAND_EDGES_HZ[1:])):
        start = np.searchsorted(right, lower, side="right")
        stop = np.searchsorted(left, upper)
        bins = slice(start, stop)
        overlap = np.minimum(upper, right[bins]) - np.maximum(lower, left[bins])
        band_energy[index] = np.dot(energy[bins], overlap / (right[bins] - left[bins]))
    return band_energy


def band_sound_pressure_level(energy_pa2_s, correction_db=0.0):
    """Normalize exposure over 70 ms and apply any selected event correction."""
    with np.errstate(divide="ignore"):
        return 10 * np.log10(energy_pa2_s / (REFERENCE_TIME_S * REFERENCE_PRESSURE_PA**2)) + correction_db


def equivalent_level(band_spl_db):
    """Evaluate the Jackson–Leventhall contours in band-number form."""
    spl = np.asarray(band_spl_db, dtype=float)
    equivalent = spl.copy()
    equivalent[26:32] -= 8
    equivalent[32:35] -= (6, 4, 2)
    equivalent[40:42] -= (4, 8)

    # Map bands below 80 Hz onto the 80 Hz contour before applying its limits.
    low_bands = BAND_NUMBERS[1:26]
    contour_band = np.maximum(low_bands, 19)
    contour_spl = spl[1:26].copy()
    contour_spl[:18] = 160 + (spl[1:19] - 160) * 19 / low_bands[:18]
    middle_offset = 1.5 * (26 - contour_band)
    lower = 76 + middle_offset
    upper = 121 + middle_offset
    scale = 26 / contour_band
    equivalent[1:26] = np.select(
        [contour_spl < lower, contour_spl > upper],
        [115 + (contour_spl - 115) * scale, 160 + (contour_spl - 160) * scale],
        default=contour_spl - middle_offset,
    ) - 8

    # The logarithmic contour is singular at 1 Hz. Use its limit from above.
    equivalent[0] = 152 if spl[0] == 160 else np.copysign(np.inf, spl[0] - 160)
    return equivalent


def level_to_sones(level_db):
    """Klos (2025), equations (2) and (4)."""
    level_db = np.asarray(level_db, dtype=float)
    quiet_ratio = 10.0 ** (np.minimum(level_db, 32) / 10)
    quiet_sones = np.cbrt(
        np.maximum(0, quiet_ratio - QUIET_PRESSURE_RATIO)
        / (ONE_SONE_PRESSURE_RATIO - QUIET_PRESSURE_RATIO)
    )
    return np.where(level_db >= 32, np.exp2((level_db - 32) / 9), quiet_sones)


def sones_to_level(total_sones):
    """Klos (2025), equations (1) and (3), with the quiet limit at zero."""
    if total_sones >= 1:
        return float(32 + 9 * np.log2(total_sones))
    pressure_ratio = (ONE_SONE_PRESSURE_RATIO - QUIET_PRESSURE_RATIO) * total_sones**3
    return float(10 * np.log10(pressure_ratio + QUIET_PRESSURE_RATIO))


def summation_factor(peak_sones):
    return float(np.interp(peak_sones, SUMMATION_SONES, SUMMATION_FACTORS))


def sum_sones(sones):
    peak_sones = float(np.max(sones))
    if np.isinf(peak_sones):
        return peak_sones
    return float(peak_sones + summation_factor(peak_sones) * (np.sum(sones) - peak_sones))


def loudness_from_spl(band_spl_db):
    equivalent = equivalent_level(band_spl_db)
    sones = level_to_sones(equivalent)
    total_sones = sum_sones(sones)
    return LoudnessSpectrum(equivalent, sones, total_sones, sones_to_level(total_sones))


def perceived_level(pressure_pa, sample_spacing_s, correction_db=0.0):
    """Calculate PL from uniformly sampled, preprocessed pressure in pascals."""
    energy = third_octave_energy(pressure_pa, sample_spacing_s)
    spl = band_sound_pressure_level(energy, correction_db)
    return PerceivedLevel(energy, spl, loudness_from_spl(spl))
