"""Check wavelet sound exposure before comparing full-record PLdB."""

import numpy as np
import pytest

from compare_mark_vii import PA_PER_PSF, waveforms
from compare_wavelet import compare_waveform
from mark_vii_reference import BAND_EDGES_HZ
from pyldb import PyLdB, WaveletExposure


def calculate_timelines(pressure_pa, dt):
    calculator = PyLdB(time=np.arange(len(pressure_pa)) * (dt * 1000), pressure=pressure_pa / PA_PER_PSF)
    calculator.perceived_loudness(method="wavelet", len_window=0, pad_front=0, pad_rear=0)
    return calculator.wavelet_exposure.exposure_rate_pa2


def test_two_sinusoids_match_the_analytic_beat_envelope_and_shift():
    sample_rate = 1024
    dt = 1 / sample_rate
    time = np.arange(sample_rate) * dt
    phase = np.pi / 3
    pressure = np.cos(2 * np.pi * 96 * time) + np.cos(2 * np.pi * 104 * time + phase)
    timelines = calculate_timelines(pressure, dt)
    shifted = calculate_timelines(np.roll(pressure, 37), dt)
    expected_exposure = np.zeros(len(BAND_EDGES_HZ) - 1)
    expected_exposure[20] = 1.0  # 100 Hz band, 0.5 Pa²·s per sinusoid.

    np.testing.assert_allclose(dt * timelines.sum(axis=1), expected_exposure, rtol=1e-12, atol=1e-24)
    np.testing.assert_allclose(shifted, np.roll(timelines, 37, axis=1), rtol=1e-12, atol=1e-14)

    # The exposure rate beats at 104 - 96 Hz. Its modulation depth is the dot
    # product of the unit gain columns, evaluated independently from the
    # direct Morlet response exp(-(omega - 6)**2/2) - exp(-18 - omega**2/2).
    modulation_depth = 0.9840268621671607
    expected_timeline = 1 + modulation_depth * np.cos(2 * np.pi * 8 * time + phase)
    np.testing.assert_allclose(timelines[20], expected_timeline, atol=1e-12, rtol=0)


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
    timelines = calculate_timelines(pressure, dt)
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


def test_calculate_wavelet_exposure_uses_processed_time_and_preserves_metrics(tmp_path):
    time_ms = 5 + np.arange(32) / 8
    pressure_psf = np.cos(2 * np.pi * 500 * time_ms / 1000) / PA_PER_PSF
    signature = tmp_path / "signature.txt"
    np.savetxt(signature, np.column_stack((time_ms, pressure_psf)))
    calculator = PyLdB(signature)
    processing = dict(len_window=4, pad_front=1, pad_rear=2)

    fft_result = calculator.calculate(**processing)
    frequency = calculator.frequency_hz.copy()
    power = calculator.power_spectrum.copy()
    results_dir = tmp_path / "results"
    wavelet_result = calculator.calculate(
        method="wavelet", print_results=True, results_dir=results_dir, **processing,
    )
    exposure = calculator.wavelet_exposure

    assert isinstance(exposure, WaveletExposure)
    np.testing.assert_array_equal(exposure.time_ms, 5 + np.arange(128) / 8)
    np.testing.assert_array_equal(exposure.time_ms, calculator.sig_time_ms)
    np.testing.assert_array_equal(exposure.band_center_hz, calculator.freq_band_center)
    assert exposure.exposure_rate_pa2.shape == (len(exposure.band_center_hz), 128)
    assert np.all(exposure.exposure_rate_pa2 >= 0)
    np.testing.assert_allclose(
        calculator.dt_s * exposure.exposure_rate_pa2.sum(axis=1),
        exposure.band_exposure_pa2_s, rtol=1e-12, atol=0,
    )
    np.testing.assert_array_equal(calculator.frequency_hz, frequency)
    np.testing.assert_array_equal(calculator.power_spectrum, power)
    np.testing.assert_array_equal(np.loadtxt(results_dir / "power_spec"), np.column_stack((frequency, power)))
    for metric in ("pldb", "ctl", "dnl", "equivalent_loudness"):
        assert getattr(wavelet_result, metric) == pytest.approx(getattr(fft_result, metric), rel=0, abs=1e-6)
    assert wavelet_result.alphabet_weighted_loudnesses == pytest.approx(fft_result.alphabet_weighted_loudnesses)


def test_wavelet_exposure_state_is_cleared_by_fft_and_signature_changes():
    time_ms = np.arange(16)
    pressure = np.ones(16)
    calculator = PyLdB(time=time_ms, pressure=pressure)
    processing = dict(len_window=0, pad_front=0, pad_rear=0)
    assert calculator.wavelet_exposure is None

    calculator.perceived_loudness(method="wavelet", **processing)
    exposure = calculator.wavelet_exposure
    with pytest.raises(ValueError, match="Method must be"):
        calculator.calculate(time=time_ms, pressure=np.zeros(16), method="unknown", **processing)
    assert calculator.wavelet_exposure is exposure
    np.testing.assert_array_equal(calculator.sig_pressure_psf, pressure)

    calculator.perceived_loudness(**processing)
    assert calculator.wavelet_exposure is None
    calculator.perceived_loudness(method="wavelet", **processing)
    calculator.set_signature(time_ms, np.zeros(16))
    assert calculator.wavelet_exposure is None
    calculator.perceived_loudness(method="wavelet", **processing)
    calculator.clear_signature()
    assert calculator.wavelet_exposure is None
