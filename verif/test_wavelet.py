"""Check wavelet sound exposure before comparing full-record PLdB."""

import numpy as np
import pytest

from compare_mark_vii import waveforms
from compare_wavelet import band_exposure_timelines, compare_waveform
from mark_vii_reference import BAND_EDGES_HZ


def test_two_sinusoids_preserve_exposure_and_shift_the_timeline():
    sample_rate = 1024
    dt = 1 / sample_rate
    time = np.arange(sample_rate) * dt
    pressure = np.cos(2 * np.pi * 96 * time) + np.cos(2 * np.pi * 104 * time)
    timelines = np.array(list(band_exposure_timelines(pressure, dt)))
    shifted = np.array(list(band_exposure_timelines(np.roll(pressure, 37), dt)))
    expected_exposure = np.zeros(len(BAND_EDGES_HZ) - 1)
    expected_exposure[20] = 1.0  # 100 Hz band, 0.5 Pa²·s per sinusoid.

    np.testing.assert_allclose(dt * timelines.sum(axis=1), expected_exposure, rtol=1e-12, atol=1e-24)
    np.testing.assert_allclose(shifted, np.roll(timelines, 37, axis=1), rtol=1e-12, atol=1e-14)
    assert np.ptp(timelines[20]) > 1.0  # The combined response varies in time.


@pytest.mark.parametrize(
    ("time_ms", "pressure_pa", "taper", "padding"),
    [pytest.param(*case, id=name) for name, *case in waveforms()],
)
def test_wavelet_matches_pyldb(time_ms, pressure_pa, taper, padding):
    result = compare_waveform(time_ms, pressure_pa, taper, padding)
    nyquist = 500 / (time_ms[1] - time_ms[0])
    above_nyquist = BAND_EDGES_HZ[:-1] >= nyquist

    np.testing.assert_allclose(
        result.wavelet_exposure_pa2_s, result.fft_exposure_pa2_s, rtol=1e-10, atol=1e-18,
    )
    np.testing.assert_array_equal(result.wavelet_exposure_pa2_s[above_nyquist], 0)
    assert result.wavelet_pldb == pytest.approx(result.fft_pldb, rel=0, abs=1e-6)


def test_zero_frequency_overlap_is_preserved_as_a_constant_timeline():
    # The DC bin of this constant record covers 0–2.5 Hz on the one-sided grid.
    dt = 1 / 80
    pressure = np.ones(16)
    timelines = np.array(list(band_exposure_timelines(pressure, dt)))
    left, right = BAND_EDGES_HZ[:-1], BAND_EDGES_HZ[1:]
    expected_rate = np.maximum(0, np.minimum(right, 2.5) - left) / 2.5
    np.testing.assert_allclose(timelines, np.broadcast_to(expected_rate[:, None], timelines.shape))

    result = compare_waveform(np.arange(16) * dt * 1000, pressure, 0, 0)
    np.testing.assert_allclose(result.wavelet_exposure_pa2_s, result.fft_exposure_pa2_s, atol=1e-18)
    assert result.wavelet_pldb == pytest.approx(result.fft_pldb, rel=0, abs=1e-6)


@pytest.mark.parametrize(("sample_count", "mean_square"), [(31, 0.5), (32, 1.0)])
def test_highest_bin_is_doubled_only_for_an_odd_record(sample_count, mean_square):
    dt = 1 / 8000
    samples = np.arange(sample_count)
    pressure = np.cos(2 * np.pi * (sample_count // 2) * samples / sample_count)
    result = compare_waveform(samples * dt * 1000, pressure, 0, 0)
    expected_exposure = np.zeros(len(BAND_EDGES_HZ) - 1)
    expected_exposure[36] = sample_count * dt * mean_square  # 4000 Hz band.
    np.testing.assert_allclose(result.wavelet_exposure_pa2_s, expected_exposure, rtol=1e-12, atol=1e-24)
    np.testing.assert_allclose(result.wavelet_exposure_pa2_s, result.fft_exposure_pa2_s, atol=1e-18)
    assert result.wavelet_pldb == pytest.approx(result.fft_pldb, rel=0, abs=1e-6)


def test_zero_pressure_has_zero_exposure_and_the_same_finite_pldb():
    with np.errstate(divide="raise", invalid="raise"):
        result = compare_waveform(np.arange(16), np.zeros(16), 0, 0)
    np.testing.assert_array_equal(result.wavelet_exposure_pa2_s, 0)
    assert result.wavelet_pldb == result.fft_pldb == pytest.approx(-3)
