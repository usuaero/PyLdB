# -*- coding: utf-8 -*-
"""Object-oriented perceived loudness calculator.

The :class:`PyLdB` class implements Stevens' Mark VII procedure for calculating
the perceived loudness of a pressure signature. Signatures are represented as
time values in milliseconds and pressure values in pounds per square foot
(psf), matching the two-column format used by ``misc/panair_r1.sig`` after its
three-line PANAIR header.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from .acoustics import alphabet_weighted_levels, dnl, equivalent, fidell_ctl


@dataclass
class LoudnessResults:
    """Container for calculated loudness metrics."""

    pldb: float
    ctl: float
    community_tolerance_level: Optional[float]
    dnl: float
    equivalent_loudness: float
    alphabet_weighted_loudnesses: Dict[str, float]

    @property
    def ctl_response(self) -> float:
        return self.ctl

    @property
    def a_weighted_loudness(self) -> float:
        return self.alphabet_weighted_loudnesses["a"]

    @property
    def b_weighted_loudness(self) -> float:
        return self.alphabet_weighted_loudnesses["b"]

    @property
    def c_weighted_loudness(self) -> float:
        return self.alphabet_weighted_loudnesses["c"]

    @property
    def d_weighted_loudness(self) -> float:
        return self.alphabet_weighted_loudnesses["d"]


class PyLdB:
    """Calculate perceived loudness for a pressure signature.

    Parameters
    ----------
    signature_file : str or pathlib.Path, optional
        Path to a two-column signature file. PANAIR ``.sig`` files can be read by
        passing ``header_lines=3``.
    time : array_like, optional
        Time samples in milliseconds.
    pressure : array_like, optional
        Pressure samples in psf.
    header_lines : int, optional
        Number of header lines to skip when ``signature_file`` is supplied.
    delimiter : str, optional
        Delimiter passed to :func:`numpy.genfromtxt`.
    """

    ref_pressure_psf = 2.900755e-9 * 144.0  # 20 micro-Pa converted to psf
    ref_time_s = 0.07

    def __init__(
        self,
        signature_file: Optional[str | Path] = None,
        *,
        time: Optional[np.ndarray] = None,
        pressure: Optional[np.ndarray] = None,
        header_lines: int = 0,
        delimiter: Optional[str] = None,
    ) -> None:
        self._initialize_table_data()
        self.clear_signature()
        self.clear_results()

        if signature_file is not None:
            self.import_signature(signature_file, header_lines=header_lines, delimiter=delimiter)
        elif time is not None or pressure is not None:
            if time is None or pressure is None:
                raise ValueError("Both time and pressure must be supplied together.")
            self.set_signature(time, pressure)

    def clear_signature(self) -> None:
        """Remove the currently loaded signature."""

        self.sig_time_ms: Optional[np.ndarray] = None
        self.sig_pressure_psf: Optional[np.ndarray] = None
        self._source_time_ms: Optional[np.ndarray] = None
        self._source_pressure_psf: Optional[np.ndarray] = None
        self.dt_ms: Optional[float] = None
        self.dt_s: Optional[float] = None
        self.freq_s: Optional[float] = None
        self.n_data_points = 0
        self.windowed_signal = False
        self.S_1 = 1.0
        self.S_1sq = 1.0
        self.S_2 = 1.0

    def clear_results(self) -> None:
        """Remove previously calculated intermediate and final results."""

        self.frequency_hz: Optional[np.ndarray] = None
        self.power_spectrum: Optional[np.ndarray] = None
        self.band_energy: Optional[np.ndarray] = None
        self.sound_pressure_level: Optional[np.ndarray] = None
        self.equivalent_loudness: Optional[np.ndarray] = None
        self.sones: Optional[np.ndarray] = None
        self.total_loudness_sones: Optional[float] = None
        self.perceived_loudness_pldb: Optional[float] = None
        self.loudness: Optional[LoudnessResults] = None

    def import_signature(
        self,
        filename: str | Path,
        *,
        header_lines: int = 0,
        delimiter: Optional[str] = None,
    ) -> "PyLdB":
        """Load a signature file into this calculator.

        The expected data section is two numeric columns: time in milliseconds
        and pressure in psf. The PANAIR sample file ``panair_r1.sig`` uses three
        text header lines, so pass ``header_lines=3`` for that format.
        """

        sig_data = np.genfromtxt(filename, skip_header=header_lines, delimiter=delimiter)
        if sig_data.ndim != 2 or sig_data.shape[1] < 2:
            raise ValueError("Signature files must contain at least two numeric columns.")
        return self.set_signature(sig_data[:, 0], sig_data[:, 1])

    def _import_sig(
        self,
        filename: str | Path,
        header_lines: int = 0,
        delimiter: Optional[str] = None,
    ) -> "PyLdB":
        """Backward-compatible alias for :meth:`import_signature`."""

        return self.import_signature(filename, header_lines=header_lines, delimiter=delimiter)

    def set_signature(self, time: np.ndarray, pressure: np.ndarray) -> "PyLdB":
        """Set the active pressure signature from arrays."""

        time_array = np.asarray(time, dtype=float)
        pressure_array = np.asarray(pressure, dtype=float)

        if time_array.ndim != 1 or pressure_array.ndim != 1:
            raise ValueError("Time and pressure must be one-dimensional arrays.")
        if len(time_array) != len(pressure_array):
            raise ValueError("Time and pressure arrays must have the same length.")
        if len(time_array) < 2:
            raise ValueError("At least two signature points are required.")

        dt = np.diff(time_array)
        if not np.all(dt > 0.0):
            raise ValueError("Time values must be strictly increasing.")
        if not np.allclose(dt, dt[0]):
            raise ValueError("Time samples must be evenly spaced.")

        self.sig_time_ms = time_array.copy()
        self.sig_pressure_psf = pressure_array.copy()
        self._source_time_ms = time_array.copy()
        self._source_pressure_psf = pressure_array.copy()
        self.dt_ms = float(dt[0])
        self.dt_s = self.dt_ms * 1.0e-3
        self.freq_s = 1.0 / self.dt_s
        self.n_data_points = len(self.sig_pressure_psf)
        self.windowed_signal = False
        self.clear_results()
        return self

    def perceived_loudness(
        self,
        time: Optional[np.ndarray] = None,
        pressure: Optional[np.ndarray] = None,
        *,
        signature_file: Optional[str | Path] = None,
        header_lines: int = 0,
        delimiter: Optional[str] = None,
        pad_front: int = 1,
        pad_rear: int = 1,
        len_window: int = 800,
        print_results: bool = False,
        results_dir: str | Path = "PyLdB_Results",
    ) -> float:
        """Calculate perceived loudness in PLdB.

        A signature may be provided at construction time, with ``set_signature``,
        with ``import_signature``, or directly to this method.
        """

        if signature_file is not None:
            self.import_signature(signature_file, header_lines=header_lines, delimiter=delimiter)
        elif time is not None or pressure is not None:
            if time is None or pressure is None:
                raise ValueError("Both time and pressure must be supplied together.")
            self.set_signature(time, pressure)
        else:
            self._require_signature()

        self._reset_working_signature()
        if pad_front < 0 or pad_rear < 0:
            raise ValueError("Padding multipliers must be non-negative.")
        if len_window < 0:
            raise ValueError("Window length must be non-negative.")

        if len_window:
            self.window(len_window)
        self.padding(n_front_points=self.n_data_points * pad_front, n_rear_points=self.n_data_points * pad_rear)

        self.frequency_hz, self.power_spectrum = self._power_spectrum()
        self.band_energy, self.sound_pressure_level = self._sound_pressure_levels(
            self.frequency_hz,
            self.power_spectrum,
        )
        self.equivalent_loudness = self._equivalent_loudness(self.sound_pressure_level)
        self.total_loudness_sones, self.sones = self._calc_total_loudness(self.equivalent_loudness)
        self.perceived_loudness_pldb = float(32.0 + 9.0 * np.log2(self.total_loudness_sones))

        if print_results:
            self.write_results(results_dir)

        return self.perceived_loudness_pldb

    def perceivedloudness(self, *args, **kwargs) -> float:
        """Backward-compatible alias for :meth:`perceived_loudness`."""

        return self.perceived_loudness(*args, **kwargs)

    def calculate(
        self,
        time: Optional[np.ndarray] = None,
        pressure: Optional[np.ndarray] = None,
        *,
        signature_file: Optional[str | Path] = None,
        header_lines: int = 0,
        delimiter: Optional[str] = None,
        pad_front: int = 1,
        pad_rear: int = 1,
        len_window: int = 800,
        ctl: Optional[float] = 75.0,
        a_star: Optional[float] = None,
        growth: float = 0.3,
        day_loudness=None,
        night_loudness=None,
        equivalent_loudnesses=None,
        print_results: bool = False,
        results_dir: str | Path = "PyLdB_Results",
    ) -> LoudnessResults:
        """Calculate PLdB and acoustic post-processing metrics.

        Results are stored on ``self.loudness`` and returned. If DNL or
        equivalent-level inputs are omitted, the calculated PLdB is used as the
        representative level.
        """

        pldb = self.perceived_loudness(
            time=time,
            pressure=pressure,
            signature_file=signature_file,
            header_lines=header_lines,
            delimiter=delimiter,
            pad_front=pad_front,
            pad_rear=pad_rear,
            len_window=len_window,
            print_results=print_results,
            results_dir=results_dir,
        )

        if day_loudness is None:
            day_loudness = pldb
        if night_loudness is None:
            night_loudness = pldb
        if equivalent_loudnesses is None:
            equivalent_loudnesses = pldb

        alphabet = alphabet_weighted_levels(self.freq_band_center, self.sound_pressure_level)
        if ctl is None and a_star is not None:
            community_tolerance_level = float(-10.0 * np.log10(-np.log(0.5)) / growth + a_star / growth)
        else:
            community_tolerance_level = ctl

        ctl_response = float(fidell_ctl(pldb, growth=growth, ctl=ctl, a_star=a_star))
        dnl_value = dnl(day_loudness, night_loudness)
        equivalent_value = equivalent(equivalent_loudnesses)

        self.loudness = LoudnessResults(
            pldb=pldb,
            ctl=ctl_response,
            community_tolerance_level=community_tolerance_level,
            dnl=dnl_value,
            equivalent_loudness=equivalent_value,
            alphabet_weighted_loudnesses=alphabet,
        )
        return self.loudness

    def window(self, len_window: int) -> "PyLdB":
        """Apply a symmetric Hann taper to the loaded pressure signature."""

        self._require_pressure()
        self.n_data_points = len(self.sig_pressure_psf)
        if len_window < 0:
            raise ValueError("Window length must be non-negative.")
        if len_window == 0:
            return self
        if 2 * len_window > self.n_data_points:
            raise ValueError("Window length cannot exceed half the signature length.")

        win = np.hanning(len_window * 2)
        self.sig_pressure_psf[:len_window] *= win[:len_window]
        self.sig_pressure_psf[-len_window:] *= win[len_window:]
        self.windowed_signal = True
        self.S_1 = float(np.sum(win))
        self.S_1sq = self.S_1**2
        self.S_2 = float(np.sum(np.square(win)))
        return self

    def padding(
        self,
        n_pad_points: Optional[int] = None,
        *,
        n_front_points: Optional[int] = None,
        n_rear_points: Optional[int] = None,
    ) -> "PyLdB":
        """Zero-pad the loaded signature and extend the time array.

        Passing ``n_pad_points`` pads both sides equally, matching the previous
        public method. ``n_front_points`` and ``n_rear_points`` may be used for
        asymmetric padding.
        """

        self._require_signature()
        if n_pad_points is not None:
            n_front_points = n_pad_points
            n_rear_points = n_pad_points
        if n_front_points is None:
            n_front_points = 0
        if n_rear_points is None:
            n_rear_points = 0
        if n_front_points < 0 or n_rear_points < 0:
            raise ValueError("Padding point counts must be non-negative.")

        n_front_points = int(n_front_points)
        n_rear_points = int(n_rear_points)
        dt = self.dt_ms

        original_time = self.sig_time_ms
        original_pressure = self.sig_pressure_psf
        front_time = np.arange(n_front_points, dtype=float) * dt
        shifted_time = original_time + n_front_points * dt
        rear_start = shifted_time[-1] + dt
        rear_time = rear_start + np.arange(n_rear_points, dtype=float) * dt

        self.sig_time_ms = np.concatenate((front_time, shifted_time, rear_time))
        self.sig_pressure_psf = np.pad(original_pressure, (n_front_points, n_rear_points), "constant")
        self.n_data_points = len(self.sig_pressure_psf)
        self.clear_results()
        return self

    def write_results(self, results_dir: str | Path = "PyLdB_Results") -> None:
        """Write signature, spectrum, and loudness intermediates to text files."""

        if self.perceived_loudness_pldb is None:
            raise RuntimeError("No calculated results are available to write.")

        directory = Path(results_dir)
        directory.mkdir(parents=True, exist_ok=True)
        np.savetxt(directory / "final_sig", np.array([self.sig_time_ms, self.sig_pressure_psf]).T)
        np.savetxt(directory / "power_spec", np.array([self.frequency_hz, self.power_spectrum]).T)
        np.savetxt(directory / "sound_pressure_levels", np.array([self.freq_band_center, self.sound_pressure_level]).T)
        np.savetxt(directory / "equivalent_loudness", np.array([self.freq_band_center, self.equivalent_loudness]).T)
        np.savetxt(directory / "sones", np.array([self.freq_band_center, self.sones]).T)

    def _initialize_table_data(self) -> None:
        base_dir = Path(__file__).resolve().parent
        tables_dir = base_dir.parent / "tables"

        equiv_loudness_sones_data = np.genfromtxt(tables_dir / "L_eq_v_sones.csv", delimiter=",", skip_header=1)
        self.equiv_loudness_sones_data = equiv_loudness_sones_data
        self.equiv_loudness_table_indep = equiv_loudness_sones_data[:, 0]
        self.sones_equiv_loudness_table = equiv_loudness_sones_data[:, 1]
        self.num_equiv_loudness_vals = len(self.equiv_loudness_table_indep)

        sones_sum_factor_data = np.genfromtxt(tables_dir / "sones_v_sumf.csv", delimiter=",", skip_header=1)
        self.sones_sum_factor_data = sones_sum_factor_data
        self.sones_table_indep = sones_sum_factor_data[:, 0]
        self.sum_factor_table = sones_sum_factor_data[:, 1]
        self.num_sum_factors = len(self.sones_table_indep)

        freq_band_data = np.genfromtxt(tables_dir / "freq_bands.csv", delimiter=",", skip_header=1)
        self.freq_band_data = freq_band_data
        self.freq_band_center = freq_band_data[:, 0]
        self.freq_band_lower_lim = freq_band_data[:, 1]
        self.freq_band_upper_lim = freq_band_data[:, 2]
        self.num_freq_bands = len(self.freq_band_center)

    def _power_spectrum(self) -> Tuple[np.ndarray, np.ndarray]:
        self._require_signature()

        # Preserve the original PyLdB FFT spacing convention. The padded time
        # vector is inclusive at both ends, so this uses duration / N rather
        # than the raw sample spacing.
        dt_s = ((self.sig_time_ms[-1] - self.sig_time_ms[0]) / self.n_data_points) * 1.0e-3
        fft_values = np.fft.rfft(self.sig_pressure_psf)
        freq = np.fft.rfftfreq(self.n_data_points, d=dt_s)
        power = np.abs(fft_values) ** 2 * dt_s**2

        if len(power) > 2:
            power[1:-1] *= 2.0
        elif len(power) == 2 and self.n_data_points % 2:
            power[1] *= 2.0

        return self._power_interp(freq, power)

    def _power_interp(self, freq: np.ndarray, power: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        interp_freq = np.append(self.freq_band_lower_lim, self.freq_band_upper_lim[-1])
        interp_power = np.interp(interp_freq, freq, power)

        full_freq = np.concatenate((freq, interp_freq))
        full_power = np.concatenate((power, interp_power))
        order = np.argsort(full_freq, kind="mergesort")
        return full_freq[order], full_power[order]

    def _sound_pressure_levels(self, freq: np.ndarray, power: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        energy = np.zeros(self.num_freq_bands)

        for j in range(self.num_freq_bands):
            section = np.nonzero((self.freq_band_lower_lim[j] <= freq) & (freq <= self.freq_band_upper_lim[j]))[0]
            if len(section) > 1:
                energy[j] = np.trapz(power[section], x=freq[section])

        energy /= self.ref_time_s
        with np.errstate(divide="ignore", invalid="ignore"):
            loudness = 10.0 * np.log10(energy / (self.ref_pressure_psf**2)) - 3.0
        return energy, loudness

    def _equivalent_loudness(self, loudness: np.ndarray) -> np.ndarray:
        L_eq = np.zeros(self.num_freq_bands)

        for i in range(self.num_freq_bands):
            if i > 39:
                L_eq[i] = loudness[i] + 4.0 * (39.0 - i)
            elif 35 <= i <= 39:
                L_eq[i] = loudness[i]
            elif 32 <= i <= 34:
                L_eq[i] = loudness[i] - 2.0 * (35.0 - i)
            elif 26 < i <= 31:
                L_eq[i] = loudness[i] - 8.0
            elif 20 <= i <= 26:
                lower_limit = 76.0 + 1.5 * (26 - i)
                upper_limit = 121.0 + 1.5 * (26 - i)
                x_value = 1.5 * (26 - i)
                L_eq[i] = self._loud_limits_400(self.freq_band_center[i], lower_limit, upper_limit, loudness[i], x_value)
            else:
                numerator = (160.0 - loudness[i]) * np.log10(80.0)
                denominator = np.log10(self.freq_band_center[i])
                L_eq_B = 160.0 - numerator / denominator
                L_eq[i] = self._loud_limits_400(80.0, 86.5, 131.5, L_eq_B, 10.5)

        return L_eq

    @staticmethod
    def _loud_limits_400(
        f_central: float,
        lower_limit: float,
        upper_limit: float,
        loudness: float,
        x_value: float,
    ) -> float:
        if loudness <= lower_limit:
            equivalent = 115.0 - ((115.0 - loudness) * np.log10(400.0)) / np.log10(f_central)
            return equivalent - 8.0
        if loudness <= upper_limit:
            return loudness - x_value - 8.0
        equivalent = 160.0 - ((160.0 - loudness) * np.log10(400.0)) / np.log10(f_central)
        return equivalent - 8.0

    def _calc_total_loudness(self, L_eq: np.ndarray) -> Tuple[float, np.ndarray]:
        sones = np.interp(
            L_eq,
            self.equiv_loudness_table_indep,
            self.sones_equiv_loudness_table,
            left=0.0,
            right=self.sones_equiv_loudness_table[-1],
        )
        max_sones = float(np.max(sones))
        sum_factor = float(
            np.interp(
                max_sones,
                self.sones_table_indep,
                self.sum_factor_table,
                left=0.0,
                right=self.sum_factor_table[-1],
            )
        )
        total_loudness = max_sones + sum_factor * (float(np.sum(sones)) - max_sones)
        return total_loudness, sones

    def _require_signature(self) -> None:
        if self.sig_time_ms is None or self.sig_pressure_psf is None:
            raise RuntimeError("No pressure signature has been loaded.")

    def _require_pressure(self) -> None:
        if self.sig_pressure_psf is None:
            raise RuntimeError("No pressure signature has been loaded.")

    def _reset_working_signature(self) -> None:
        if self._source_time_ms is None or self._source_pressure_psf is None:
            return

        self.sig_time_ms = self._source_time_ms.copy()
        self.sig_pressure_psf = self._source_pressure_psf.copy()
        self.dt_ms = float(self.sig_time_ms[1] - self.sig_time_ms[0])
        self.dt_s = self.dt_ms * 1.0e-3
        self.freq_s = 1.0 / self.dt_s
        self.n_data_points = len(self.sig_pressure_psf)
        self.windowed_signal = False
        self.clear_results()


def import_sig(filename: str | Path, header_lines: int = 0, delimiter: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Load a pressure signature and return ``(time_ms, pressure_psf)`` arrays."""

    calculator = PyLdB()
    calculator.import_signature(filename, header_lines=header_lines, delimiter=delimiter)
    return calculator.sig_time_ms.copy(), calculator.sig_pressure_psf.copy()


def perceivedloudness(
    time: np.ndarray,
    pressure: np.ndarray,
    *,
    pad_front: int = 1,
    pad_rear: int = 1,
    len_window: int = 800,
    print_results: bool = False,
) -> float:
    """Backward-compatible function wrapper around :class:`PyLdB`."""

    return PyLdB(time=time, pressure=pressure).perceived_loudness(
        pad_front=pad_front,
        pad_rear=pad_rear,
        len_window=len_window,
        print_results=print_results,
    )


__all__ = ["LoudnessResults", "PyLdB", "import_sig", "perceivedloudness"]
