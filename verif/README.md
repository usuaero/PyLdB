# Stevens Mark VII verification

This report compares the current PyLdB calculation with an independent implementation of the published Mark VII equations and the 2025 NASA recommendations. [Stevens, 1972](https://doi.org/10.1121/1.1912880) · [NASA TM-20250003346](https://ntrs.nasa.gov/citations/20250003346)

## Reproduce

With PyLdB and pytest installed, run these commands from the repository root.

```sh
pytest --import-mode=importlib -o pythonpath=verif verif/test.py verif/test_mark_vii.py
python verif/compare_mark_vii.py
```

The comparison prints exposure, SPL, equivalent level, and sones for the loudest bands, as well as the total sones and PLdB. The tests cover FFT energy conservation for odd and even record lengths, the band-edge split, zero energy above Nyquist, and the analytic spectrum of the NASA N-wave. Both implementations are also checked against NASA's three published quiet spectra with a 0.003 PLdB tolerance to account for rounding of the input SPLs. [NASA TM-20250003346, Table 2](https://ntrs.nasa.gov/citations/20250003346)

## Method and inputs

- Both calculations use the same processed samples. They allocate one-sided FFT bin energy by overlap with exact base-10 one-third-octave bands. The reference works in pascals, and the comparison converts PyLdB's stored mean-square pressure back to exposure in Pa² s. Both calculations normalize exposure over 70 ms with a 20 µPa reference pressure. [NASA TM-20250003346, Appendix 1](https://ntrs.nasa.gov/citations/20250003346) · [PCBoom Technical Reference, 2nd ed.](https://ntrs.nasa.gov/citations/20250003228)
- The loudness stage uses the Jackson–Leventhall band-number equations, including the limiting contour at 1 Hz, and Klos's piecewise level and sone conversions. Stevens' summation table includes the two quiet-level points recommended by NASA. [Bolander et al., 2019, Appendix A](https://doi.org/10.2514/6.2019-2091) · [NASA TM-20250003346, Sections II.C–D](https://ntrs.nasa.gov/citations/20250003346)
- PANAIR and its 1%, 0.4%, and 0.1% pressure-scaled variants use an 800-sample Hann taper at each end and six record lengths of zero padding on each side. The 48 kHz NASA N-wave has 50 Pa peak overpressure, an 8 ms rise time, and a 350 ms duration. It starts and ends at zero and uses three record lengths of padding on each side. The untapered periodic probes use 1 Pa peak tones at 3162 Hz and 89 Hz in one-second, 32 kHz records. The 8 kHz case combines a 1 Pa peak 500 Hz cosine with a 0.01 Pa Nyquist component over 512 samples. [PANAIR fixture](../misc/panair_r1.sig) · [NASA TP-3134, Figures 1–2 and Eq. (1)](https://ntrs.nasa.gov/citations/19920002547)
- The comparison applies the same −3 dB band adjustment to both calculations. The script also prints an unadjusted reference result to show the effect of the two-pulse convention. [Bolander et al., 2019, Eqs. (9)–(10)](https://doi.org/10.2514/6.2019-2091)

## Results

All eight cases agree within 0.000002 PLdB with the matched adjustment. The signed difference is PyLdB minus the reference. PyLdB assigns zero energy to bands entirely above Nyquist in the 8 kHz case, and both calculations return −3 PLdB at zero sones.

| Waveform | PyLdB (PLdB) | Reference (PLdB) | Difference (PLdB) |
|---|---:|---:|---:|
| PANAIR | 77.70673 | 77.70673 | −8.304e−7 |
| PANAIR at 1.0% amplitude | 28.15343 | 28.15343 | −1.145e−6 |
| PANAIR at 0.4% amplitude | 12.14730 | 12.14730 | −1.682e−6 |
| PANAIR at 0.1% amplitude | −3.00000 | −3.00000 | 0 |
| NASA N-wave | 89.80950 | 89.80950 | −8.350e−7 |
| Reference-band tone | 99.51812 | 99.51812 | −7.348e−7 |
| Band-edge tone | 81.12388 | 81.12388 | −7.348e−7 |
| 8 kHz sampling | 79.84261 | 79.84261 | −7.358e−7 |

See the [changelog](../CHANGELOG.md) for changes in the reported levels.
