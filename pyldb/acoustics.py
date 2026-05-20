"""Acoustic post-processing utilities."""

from __future__ import annotations

import numpy as np


def fidell_ctl(noise, growth=0.3, ctl=None, a_star=None):
    """Calculate Fidell CTL high-annoyance response probability."""

    if ctl is None and a_star is None:
        raise ValueError("Either ctl or a_star must be supplied.")

    if ctl is None:
        ctl = -10.0 * np.log10(-np.log(0.5)) / growth + a_star / growth
        m_ctl = (10.0 ** (ctl / 10.0)) ** growth
        a_value = -m_ctl * np.log(0.5)
    else:
        a_star = growth * ctl + 10.0 * np.log10(-np.log(0.5))
        a_value = 10.0 ** (a_star / 10.0)

    m_noise = (10.0 ** (np.asarray(noise, dtype=float) / 10.0)) ** growth
    return np.exp(-a_value / m_noise)


def fidell_CTL(noise, growth=0.3, CTL=None, A_star=None):
    """Backward-compatible wrapper for :func:`fidell_ctl`."""

    return fidell_ctl(noise, growth=growth, ctl=CTL, a_star=A_star)


def dnl(day_loudness, night_loudness):
    """Calculate day-night average level from 15 day and 9 night hourly values."""

    day = _as_hour_array(day_loudness, 15, "day_loudness")
    night = _as_hour_array(night_loudness, 9, "night_loudness")

    energy = np.sum(10.0 ** (day / 10.0))
    energy += np.sum(10.0 * 10.0 ** (night / 10.0))
    return float(10.0 * np.log10(energy / 24.0))


def DNL(Lh_day, Lh_night):
    """Backward-compatible wrapper for :func:`dnl`."""

    return dnl(Lh_day, Lh_night)


def equivalent(loudnesses, sampling=None):
    """Calculate equivalent level for a scalar or sequence of levels."""

    values = np.asarray(loudnesses, dtype=float)
    if values.ndim == 0:
        if sampling is None:
            sampling = 1
        values = np.full(int(sampling), float(values))
    if values.size == 0:
        raise ValueError("At least one loudness value is required.")

    return float(10.0 * np.log10(np.mean(10.0 ** (values / 10.0))))


def alphabet_weighted_levels(frequencies_hz, levels_db):
    """Calculate overall A-, B-, C-, and D-weighted levels from band levels."""

    frequencies = np.asarray(frequencies_hz, dtype=float)
    levels = np.asarray(levels_db, dtype=float)
    if frequencies.shape != levels.shape:
        raise ValueError("Frequency and level arrays must have the same shape.")

    return {
        "a": _weighted_overall(levels, a_weighting(frequencies)),
        "b": _weighted_overall(levels, b_weighting(frequencies)),
        "c": _weighted_overall(levels, c_weighting(frequencies)),
        "d": _weighted_overall(levels, d_weighting(frequencies)),
    }


def a_weighting(frequencies_hz):
    """Return A-weighting adjustments in dB."""

    f = np.asarray(frequencies_hz, dtype=float)
    f2 = f**2
    numerator = (12200.0**2) * f2**2
    denominator = (f2 + 20.6**2) * np.sqrt((f2 + 107.7**2) * (f2 + 737.9**2)) * (f2 + 12200.0**2)
    return 2.0 + 20.0 * np.log10(numerator / denominator)


def b_weighting(frequencies_hz):
    """Return B-weighting adjustments in dB."""

    f = np.asarray(frequencies_hz, dtype=float)
    f2 = f**2
    numerator = (12200.0**2) * f**3
    denominator = (f2 + 20.6**2) * np.sqrt(f2 + 158.5**2) * (f2 + 12200.0**2)
    return 0.17 + 20.0 * np.log10(numerator / denominator)


def c_weighting(frequencies_hz):
    """Return C-weighting adjustments in dB."""

    f = np.asarray(frequencies_hz, dtype=float)
    f2 = f**2
    numerator = (12200.0**2) * f2
    denominator = (f2 + 20.6**2) * (f2 + 12200.0**2)
    return 0.06 + 20.0 * np.log10(numerator / denominator)


def d_weighting(frequencies_hz):
    """Return D-weighting adjustments in dB."""

    f = np.asarray(frequencies_hz, dtype=float)
    h = ((1037918.48 - f**2) ** 2 + 1080768.16 * f**2) / ((9837328.0 - f**2) ** 2 + 11723776.0 * f**2)
    numerator = f / 6.8966888496476e-5 * np.sqrt(h / ((f**2 + 79919.29) * (f**2 + 1345600.0)))
    return 20.0 * np.log10(numerator)


def _weighted_overall(levels_db, adjustments_db):
    finite = np.isfinite(levels_db)
    if not np.any(finite):
        return float("-inf")
    weighted = levels_db[finite] + adjustments_db[finite]
    return float(10.0 * np.log10(np.sum(10.0 ** (weighted / 10.0))))


def _as_hour_array(values, expected_length, name):
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        return np.full(expected_length, float(array))
    if len(array) != expected_length:
        raise ValueError(f"{name} must be scalar or contain {expected_length} hourly values.")
    return array


__all__ = [
    "DNL",
    "a_weighting",
    "alphabet_weighted_levels",
    "b_weighting",
    "c_weighting",
    "d_weighting",
    "dnl",
    "equivalent",
    "fidell_CTL",
    "fidell_ctl",
]
