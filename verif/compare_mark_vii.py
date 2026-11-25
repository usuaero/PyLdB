"""Print the reproducible Mark VII comparison as Markdown."""

from pathlib import Path

import numpy as np

import mark_vii_reference as reference
from pyldb import PyLdB


PA_PER_PSF = 4.4482216152605 / 0.3048**2
PANAIR_PATH = Path(__file__).resolve().parents[1] / "misc" / "panair_r1.sig"


def nasa_n_wave():
    """Sample the N-wave in NASA TP-3134, Figures 1–2, at 48 kHz."""
    time_s = np.arange(0, 0.5, 1 / 48000)
    pressure_pa = np.interp(
        time_s, [0.075, 0.083, 0.417, 0.425], [0, 50, -50, 0], left=0, right=0,
    )
    return time_s, pressure_pa


def waveforms():
    panair_time_ms, panair_pressure_psf = np.loadtxt(PANAIR_PATH, skiprows=3).T
    for amplitude_scale in (1, 0.01, 0.004, 0.001):
        name = "PANAIR" if amplitude_scale == 1 else f"PANAIR at {amplitude_scale:.1%} amplitude"
        pressure_pa = panair_pressure_psf * (PA_PER_PSF * amplitude_scale)
        yield name, panair_time_ms, pressure_pa, 800, 6

    time_s, wave = nasa_n_wave()
    yield "NASA N-wave", time_s * 1000, wave, 0, 3

    # Periodic records make the spectral allocation of these tones reproducible.
    time_s = np.arange(32000) / 32000
    for frequency, name in [(3162, "Reference-band tone"), (89, "Band-edge tone")]:
        yield name, time_s * 1000, np.cos(2 * np.pi * frequency * time_s), 0, 0

    sample = np.arange(512)
    pressure_pa = np.cos(2 * np.pi * 500 * sample / 8000) + 0.01 * (-1.0) ** sample
    yield "8 kHz sampling", sample / 8, pressure_pa, 0, 0


def compare_waveforms():
    summaries = []
    stages = []
    bands = []
    for name, time_ms, pressure_pa, taper, padding in waveforms():
        calculator = PyLdB(time=time_ms, pressure=pressure_pa / PA_PER_PSF)
        # For zero sones the production logarithm returns -inf, reported below.
        with np.errstate(divide="ignore"):
            production_pl = calculator.perceived_loudness(
                len_window=taper, pad_front=padding, pad_rear=padding,
            )
        spacing = (time_ms[1] - time_ms[0]) / 1000
        energy = reference.third_octave_energy(calculator.sig_pressure_psf * PA_PER_PSF, spacing)
        unadjusted = reference.loudness_from_spl(reference.band_sound_pressure_level(energy))
        ref_spl = reference.band_sound_pressure_level(energy, correction_db=-3)
        matched = reference.loudness_from_spl(ref_spl)
        summaries.append(
            f"| {name} | {production_pl:.5f} | {matched.pldb:.5f} | {production_pl - matched.pldb:+.5f}"
            f" | {unadjusted.pldb:.5f} | {production_pl - unadjusted.pldb:+.5f} |"
        )

        # Substitute each PyLdB stage in order while holding downstream math fixed.
        after_spl = reference.loudness_from_spl(calculator.sound_pressure_level).pldb
        after_contour = reference.sones_to_level(
            reference.sum_sones(reference.level_to_sones(calculator.equivalent_loudness))
        )
        after_band_sones = reference.sones_to_level(reference.sum_sones(calculator.sones))
        after_summation = reference.sones_to_level(calculator.total_loudness_sones)
        stages.append(
            f"| {name} | {after_spl - matched.pldb:+.5f} | {after_contour - after_spl:+.5f}"
            f" | {after_band_sones - after_contour:+.5f} | {after_summation - after_band_sones:+.5f}"
            f" | {production_pl - after_summation:+.5f} |"
        )

        production_energy = calculator.band_energy * (0.070 * PA_PER_PSF**2)
        above_nyquist = reference.BAND_EDGES_HZ[:-1] >= 0.5 / spacing
        bands.append(
            f"| {name} | {production_energy.sum():.6g} | {energy.sum():.6g}"
            f" | {production_energy[above_nyquist].sum():.6g}"
            f" | {calculator.total_loudness_sones:.6f} | {matched.total_sones:.6f} |"
        )
        loudest_bands = {
            int(np.argmax(sones)) for sones in (matched.sones, calculator.sones) if np.any(sones)
        }
        for index in sorted(loudest_bands):
            center_hz = reference.BAND_CENTERS_HZ[index]
            yield (
                f"{name}, {center_hz:.2f} Hz loudest band in one or both methods "
                f"(PyLdB/reference with -3 dB)\n"
                f"  Energy: {production_energy[index]:.6g}/{energy[index]:.6g} Pa^2 s\n"
                f"  SPL: {calculator.sound_pressure_level[index]:.5f}/{ref_spl[index]:.5f} dB\n"
                f"  Equivalent level: {calculator.equivalent_loudness[index]:.5f}/"
                f"{matched.equivalent_level_db[index]:.5f} dB\n"
                f"  Sones: {calculator.sones[index]:.6f}/{matched.sones[index]:.6f}"
            )

    yield "\n| Waveform | PyLdB | Reference -3 dB | Delta, matched | Reference 0 dB | Delta, unadjusted |"
    yield "|---|---:|---:|---:|---:|---:|"
    yield from summaries
    yield "\n| Waveform | Band SPL | Contours | Band sones | Summation | Final conversion |"
    yield "|---|---:|---:|---:|---:|---:|"
    yield from stages
    yield "\n| Waveform | PyLdB band energy | Reference band energy | PyLdB above Nyquist | PyLdB sones | Reference sones |"
    yield "|---|---:|---:|---:|---:|---:|"
    yield from bands


if __name__ == "__main__":
    for line in compare_waveforms():
        print(line)
