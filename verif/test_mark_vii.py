"""Analytical and published checks for the independent Mark VII reference."""

import math

import numpy as np
import pytest

import mark_vii_reference as reference
from compare_mark_vii import PA_PER_PSF, nasa_n_wave
from pyldb import PyLdB


def test_mark_vii_reference_band():
    band_spl = np.full(42, -np.inf)
    band_spl[35] = 32.0  # Nominal 3150 Hz band.
    result = reference.loudness_from_spl(band_spl)

    assert result.total_sones == 1
    assert result.pldb == 32

    calculator = PyLdB()
    equivalent = calculator._equivalent_loudness(band_spl)
    total_sones, sones = calculator._calc_total_loudness(equivalent)
    assert total_sones == result.total_sones
    np.testing.assert_array_equal(sones, result.sones)


def test_two_one_sone_bands_use_stevens_summation_factor():
    band_spl = np.full(42, -np.inf)
    band_spl[35:37] = 32.0
    result = reference.loudness_from_spl(band_spl)
    assert result.total_sones == pytest.approx(1.316)  # Stevens (1972), p. 597.
    assert result.pldb == pytest.approx(32 + 9 * math.log2(1.316))

    calculator = PyLdB()
    total_sones, _ = calculator._calc_total_loudness(calculator._equivalent_loudness(band_spl))
    assert total_sones == pytest.approx(result.total_sones)


# Klos (2025), Table 2. The 1 Hz band is omitted in the published spectra.
@pytest.mark.parametrize(
    ("spectrum_sones", "expected_pldb", "published_spl"),
    [
        (0.180, 30.472, [
            153.8964, 147.7928, 141.6892, 135.5856, 129.4821, 123.3785,
            117.2749, 111.1713, 105.0677, 98.96412, 92.86054, 86.75695,
            80.65336, 74.54977, 68.44619, 62.34260, 56.23901, 50.13542,
            44.03184, 40.29667, 36.56150, 32.82634, 29.09117, 25.35600,
            21.62084, *([17.88567] * 6), 15.88567, 13.88567, 11.88567,
            *([9.88567] * 5), 13.88567, 17.88567,
        ]),
        (0.181, 30.700, [
            153.8991, 147.7981, 141.6972, 135.5963, 129.4954, 123.3944,
            117.2935, 111.1926, 105.0917, 98.99074, 92.88981, 86.78888,
            80.68796, 74.58703, 68.48610, 62.38518, 56.28425, 50.18332,
            44.08240, 40.34989, 36.61739, 32.88488, 29.15238, 25.41987,
            21.68736, *([17.95486] * 6), 15.95486, 13.95486, 11.95486,
            *([9.95486] * 5), 13.95486, 17.95486,
        ]),
        (0.182, 30.922, [
            153.9017, 147.8033, 141.7050, 135.6067, 129.5083, 123.4100,
            117.3117, 111.2134, 105.1150, 99.01670, 92.91837, 86.82004,
            80.72171, 74.62338, 68.52505, 62.42672, 56.32839, 50.23006,
            44.13173, 40.40182, 36.67191, 32.94200, 29.21209, 25.48218,
            21.75227, *([18.02236] * 6), 16.02236, 14.02236, 12.02236,
            *([10.02236] * 5), 14.02236, 18.02236,
        ]),
    ],
)
def test_nasa_quiet_spectra(spectrum_sones, expected_pldb, published_spl):
    result = reference.loudness_from_spl(np.array([-np.inf, *published_spl]))
    assert result.sones[0] == 0
    # The least precise printed SPLs occur in the steepest low-frequency contours.
    np.testing.assert_allclose(result.sones[1:], spectrum_sones, rtol=0, atol=2e-5)
    assert result.pldb == pytest.approx(expected_pldb, abs=0.003)


def test_nasa_quiet_summation_breakpoints():
    assert reference.summation_factor(0) == 0
    assert reference.summation_factor(0.113) == 0
    assert reference.summation_factor(0.147) == pytest.approx(0.050)
    assert reference.summation_factor(0.181) == pytest.approx(0.100)
    assert reference.summation_factor(0.196) == pytest.approx(0.122)
    assert reference.sones_to_level(0) == pytest.approx(-3)
    assert reference.level_to_sones(-3) == pytest.approx(0)


@pytest.mark.parametrize("sample_count", [31, 32])
def test_fft_preserves_energy_in_highest_positive_bin(sample_count):
    spacing = 1 / 32000
    sample = np.arange(sample_count)
    pressure = 0.2 + np.cos(2 * np.pi * (sample_count // 2) * sample / sample_count)
    _, energy = reference.fft_bin_energy(pressure, spacing)

    assert energy.sum() == pytest.approx(spacing * np.dot(pressure, pressure), rel=1e-12)


def test_third_octave_bands_above_nyquist_are_empty():
    spacing = 1 / 8000
    pressure = (-1.0) ** np.arange(512)
    energy = reference.third_octave_energy(pressure, spacing)
    above_nyquist = reference.BAND_EDGES_HZ[:-1] >= 0.5 / spacing

    np.testing.assert_array_equal(energy[above_nyquist], 0)
    assert energy.sum() == pytest.approx(spacing * np.dot(pressure, pressure), rel=1e-12)


def test_fft_bin_straddling_a_band_edge_splits_energy_by_overlap():
    samples = np.arange(32000)
    pressure = np.cos(2 * np.pi * 89 * samples / len(samples))
    energy = reference.third_octave_energy(pressure, 1 / 32000)
    # The 89 Hz bin spans [88.5, 89.5] Hz, with the band edge inside.
    edge = 10.0 ** 1.95
    expected_energy = 0.5  # A 1 Pa peak cosine over one second.

    assert energy[19] == pytest.approx((edge - 88.5) * expected_energy, rel=1e-12)
    assert energy[20] == pytest.approx((89.5 - edge) * expected_energy, rel=1e-12)
    assert energy.sum() == pytest.approx(expected_energy, rel=1e-12)


def test_bin_centered_reference_band_tone_end_to_end():
    sample_rate = 32000
    samples = np.arange(sample_rate)
    # For this one-second record, 2 * target exposure gives peak pressure squared.
    amplitude_pa = 20e-6 * math.sqrt(2 * 0.070 * 10**3.2)
    pressure = amplitude_pa * np.cos(2 * np.pi * 3162 * samples / sample_rate)
    result = reference.perceived_level(pressure, 1 / sample_rate)

    assert result.band_spl_db[35] == pytest.approx(32, abs=1e-10)
    assert result.loudness.pldb == pytest.approx(32, abs=1e-10)


def test_pyldb_reference_band_tone_matches_reference_with_same_adjustment():
    sample_rate = 32000
    time_s = np.arange(sample_rate) / sample_rate
    pressure_pa = np.cos(2 * np.pi * 3162 * time_s)
    exact = reference.perceived_level(pressure_pa, 1 / sample_rate, correction_db=-3)
    calculator = PyLdB(time=time_s * 1000, pressure=pressure_pa / PA_PER_PSF)

    actual = calculator.perceived_loudness(pad_front=0, pad_rear=0, len_window=0)

    assert actual == pytest.approx(exact.loudness.pldb, abs=0.01)


def test_nasa_n_wave_against_published_analytic_spectrum():
    time_s, pressure_pa = nasa_n_wave()
    dt = time_s[1] - time_s[0]
    frequencies, energy = reference.fft_bin_energy(pressure_pa, dt)
    frequency_indices = np.array([5, 25, 50])  # 10, 50, and 100 Hz.
    angular_frequency = 2 * np.pi * frequencies[frequency_indices]
    # NASA TP-3134 Eq. (1), with T2 the half-duration and T2 - T1 the rise time.
    t2 = 0.175
    t1 = t2 - 0.008
    transform = (
        2 * 50 * (t2 * np.sin(angular_frequency * t1) - t1 * np.sin(angular_frequency * t2))
        / (angular_frequency**2 * t1 * (t2 - t1))
    )
    expected_bin_energy = 2 * transform**2 / (len(pressure_pa) * dt)
    np.testing.assert_allclose(energy[frequency_indices], expected_bin_energy, rtol=4e-5)
