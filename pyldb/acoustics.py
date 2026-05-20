"""Acoustic post-processing utilities.

This module contains helper methods to process calculated loudness levels into
community response, day-night average level, energy-equivalent loudness, and
standard alphabet-weighted overall levels.

References
----------
Fidell, S., et al., "A first-principles model for estimating the prevalence of
annoyance with aircraft noise exposure," The Journal of the Acoustical Society
of America, Vol. 130, 791, 2011.

Fidell, S., et al., "Community Response to High-Energy Impulsive Sounds: An
Assessment of the Field Since 1981," National Research Council, 1996.
"""

from __future__ import annotations

import numpy as np


def fidell_ctl(noise, growth=0.3, ctl=None, a_star=None):
    """Calculate Fidell CTL high-annoyance response probability.

    This function evaluates the Fidell community response model for a noise
    level or array of noise levels. A Community Tolerance Level (CTL) can be
    supplied directly, or it can be derived from ``a_star`` and ``growth``.

    Parameters
    ----------
    noise : float or array_like
        Noise level or levels, in decibels.
    growth : float, optional
        Growth rate for the response curve. A value of 0.3 is commonly used for
        subsonic aircraft noise, while larger values may be used for other
        contexts.
    ctl : float, optional
        Community Tolerance Level.
    a_star : float, optional
        Fidell model intercept. Required when ``ctl`` is not supplied.

    Returns
    -------
    float or numpy.ndarray
        High-annoyance response probability for the supplied noise level or
        levels.
    """

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
    """Calculate day-night average level (DNL).

    DNL is calculated from 15 daytime hourly levels and 9 nighttime hourly
    levels. Nighttime levels receive the standard 10 dB penalty before the
    24-hour energy average is calculated.

    Parameters
    ----------
    day_loudness : float or array_like
        Daytime hourly equivalent loudness values. A scalar is repeated for all
        15 daytime hours; an array must contain 15 values.
    night_loudness : float or array_like
        Nighttime hourly equivalent loudness values. A scalar is repeated for
        all 9 nighttime hours; an array must contain 9 values.

    Returns
    -------
    float
        Day-night average level.
    """

    day = _as_hour_array(day_loudness, 15, "day_loudness")
    night = _as_hour_array(night_loudness, 9, "night_loudness")

    energy = np.sum(10.0 ** (day / 10.0))
    energy += np.sum(10.0 * 10.0 ** (night / 10.0))
    return float(10.0 * np.log10(energy / 24.0))


def DNL(Lh_day, Lh_night):
    """Backward-compatible wrapper for :func:`dnl`."""

    return dnl(Lh_day, Lh_night)


def equivalent(loudnesses, sampling=None):
    """Calculate energy-equivalent loudness.

    Equivalent loudness is calculated by converting each level to energy,
    averaging those energies, and converting the result back to decibels. If a
    scalar loudness is supplied, ``sampling`` may be used to repeat that value.

    Parameters
    ----------
    loudnesses : float or array_like
        Loudness value or values to average.
    sampling : int, optional
        Number of repeated samples to use when ``loudnesses`` is scalar.

    Returns
    -------
    float
        Energy-equivalent loudness level.
    """

    values = np.asarray(loudnesses, dtype=float)
    if values.ndim == 0:
        if sampling is None:
            sampling = 1
        values = np.full(int(sampling), float(values))
    if values.size == 0:
        raise ValueError("At least one loudness value is required.")

    return float(10.0 * np.log10(np.mean(10.0 ** (values / 10.0))))


def alphabet_weighted_levels(frequencies_hz, levels_db):
    """Calculate overall A-, B-, C-, and D-weighted levels.

    Parameters
    ----------
    frequencies_hz : array_like
        Frequency band centers in hertz.
    levels_db : array_like
        Unweighted band sound pressure levels in decibels.

    Returns
    -------
    dict
        Overall weighted levels keyed as ``"a"``, ``"b"``, ``"c"``, and
        ``"d"``.
    """

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
    """Return A-weighting adjustments in dB.

    Parameters
    ----------
    frequencies_hz : array_like
        Frequencies in hertz.

    Returns
    -------
    numpy.ndarray
        A-weighting adjustments in decibels.
    """

    f = np.asarray(frequencies_hz, dtype=float)
    f2 = f**2
    numerator = (12200.0**2) * f2**2
    denominator = (f2 + 20.6**2) * np.sqrt((f2 + 107.7**2) * (f2 + 737.9**2)) * (f2 + 12200.0**2)
    return 2.0 + 20.0 * np.log10(numerator / denominator)


def b_weighting(frequencies_hz):
    """Return B-weighting adjustments in dB.

    Parameters
    ----------
    frequencies_hz : array_like
        Frequencies in hertz.

    Returns
    -------
    numpy.ndarray
        B-weighting adjustments in decibels.
    """

    f = np.asarray(frequencies_hz, dtype=float)
    f2 = f**2
    numerator = (12200.0**2) * f**3
    denominator = (f2 + 20.6**2) * np.sqrt(f2 + 158.5**2) * (f2 + 12200.0**2)
    return 0.17 + 20.0 * np.log10(numerator / denominator)


def c_weighting(frequencies_hz):
    """Return C-weighting adjustments in dB.

    Parameters
    ----------
    frequencies_hz : array_like
        Frequencies in hertz.

    Returns
    -------
    numpy.ndarray
        C-weighting adjustments in decibels.
    """

    f = np.asarray(frequencies_hz, dtype=float)
    f2 = f**2
    numerator = (12200.0**2) * f2
    denominator = (f2 + 20.6**2) * (f2 + 12200.0**2)
    return 0.06 + 20.0 * np.log10(numerator / denominator)


def d_weighting(frequencies_hz):
    """Return D-weighting adjustments in dB.

    Parameters
    ----------
    frequencies_hz : array_like
        Frequencies in hertz.

    Returns
    -------
    numpy.ndarray
        D-weighting adjustments in decibels.
    """

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
