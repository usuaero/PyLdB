# Stevens Mark VII verification

This report compares the current PyLdB calculation with an independent implementation of the published Mark VII equations and the 2025 NASA recommendations. [Stevens, 1972](https://doi.org/10.1121/1.1912880) · [NASA TM-20250003346](https://ntrs.nasa.gov/citations/20250003346)

## Reproduce

With PyLdB and pytest installed, run these commands from the repository root.

```sh
pytest --import-mode=importlib -o pythonpath=verif verif/test.py verif/test_mark_vii.py verif/test_wavelet.py
python verif/compare_mark_vii.py
python verif/compare_wavelet.py
```

The Mark VII comparison prints exposure, SPL, equivalent level, and sones for the loudest bands, as well as the total sones and PLdB. The tests cover FFT energy conservation for odd and even record lengths, the band-edge split, zero energy above Nyquist, and the analytic spectrum of the NASA N-wave. The tests check PyLdB and the independent Mark VII reference against NASA's three published quiet spectra with a 0.003 PLdB tolerance to account for rounding of the input SPLs. [NASA TM-20250003346, Table 2](https://ntrs.nasa.gov/citations/20250003346)

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

## Wavelet verification against PyLdB 1.0.0

Band sound exposure is the integral of squared band pressure over time, in Pa²·s. A band exposure timeline gives the exposure rate at each sample, in Pa². Multiply the sum of the timeline samples by the sample interval to recover the exposure. Sound exposure level expresses exposure relative to a reference in decibels. PLdB is the perceived level from the Mark VII calculation. [ASA/ANSI sound exposure](https://asastandards.org/terms/sound-exposure/) · [ASA/ANSI sound exposure level](https://asastandards.org/terms/sound-exposure-level/)

### From frequency bins to a timeline

1. Assign exposure to a band. Both calculations start with the same tapered and padded pressure record. An FFT bin is one frequency sample that represents a small frequency interval. Each third-octave band receives the fraction of the bin's interval that overlaps the band. If that fraction is 60%, multiply the complex bin amplitude by $\sqrt{0.60}$. Its squared magnitude then drops to 60% of the original, assigning 60% of the bin's exposure to that band while preserving phase.
2. Split the band coefficients among six wavelets. Start with six Morlet-shaped frequency responses whose center frequencies are spaced logarithmically within the band. Evaluate each wavelet's gain at each participating FFT frequency. In this matrix, each column is a frequency bin and each row is a wavelet. Divide each column by the square root of the sum of its squared gains so the six squared gains sum to one. The squared gain is the wavelet's share of the exposure assigned to that bin in this band. The gain matrix depends on the FFT grid and the chosen wavelets. The measured FFT exposure totals are used only for comparison.
3. Return to the time axis. Multiply the band's complex FFT coefficients by each row of gains and take one inverse FFT per wavelet. The new matrix contains wavelet coefficients. Each row is still a wavelet, but each column is now a sample time. At each time, sum the squared magnitudes down the column to get one exposure rate for the band.
4. Recover full-record exposure and PLdB. Parseval's identity equates the sum of squared magnitudes in time to the corresponding sum in frequency, with FFT scaling accounted for. Since the squared gains at every bin sum to one, integrating the timeline over the record recovers exactly the band's assigned exposure in exact arithmetic. Apply the same 70 ms normalization, −3 dB band adjustment, and Mark VII calculation as the baseline.

The inverse transforms use just the one-sided spectrum. Multiplying positive-frequency bins with a negative-frequency partner by $\sqrt{2}$ restores their two-sided exposure. The even-length Nyquist bin has no separate partner and is counted once. The Morlet wavelets have zero gain at zero frequency, so any assigned exposure from the zero-frequency bin is carried as a constant rate over the record.

### Two-sinusoid example

A one-second pressure record contains 96 Hz and 104 Hz sinusoids, each with a peak amplitude of 1 Pa. The 104 Hz sinusoid starts with a 60° phase offset. Both FFT bins lie entirely in the nominal 100 Hz band. Each sinusoid contributes 0.5 Pa²·s, so the band contains 1.0 Pa²·s. The six wavelets split each bin's exposure among their responses.

A beat is the repeating rise and fall that occurs when two sinusoids of nearby frequencies alternately reinforce and partly cancel each other. Here the frequencies differ by 8 Hz, so the exposure timeline repeats every 125 ms. The test checks the rate at each sample against the expected beat, including its phase and amplitude. The average rate is 1 Pa², so its integral is still 1.0 Pa²·s.

### Results

All eight waveforms match the FFT baseline to eight decimal places in PLdB. Across all cases, the largest absolute difference in any band's sound exposure was 7.105 × 10⁻¹⁵ Pa²·s in this run.

| Waveform | FFT PLdB | Wavelet PLdB |
|---|---:|---:|
| PANAIR | 77.70672980 | 77.70672980 |
| PANAIR at 1.0% amplitude | 28.15342894 | 28.15342894 |
| PANAIR at 0.4% amplitude | 12.14729549 | 12.14729549 |
| PANAIR at 0.1% amplitude | −3.00000000 | −3.00000000 |
| NASA N-wave | 89.80950163 | 89.80950163 |
| Reference-band tone | 99.51811900 | 99.51811900 |
| Band-edge tone | 81.12387761 | 81.12387761 |
| 8 kHz sampling | 79.84261149 | 79.84261149 |

The analytical checks verify the two-sinusoid exposure and beat, the timeline's response to a circular shift, and the treatment of zero pressure, zero frequency, and the highest bins of odd and even records. The FFT treats the processed record as periodic, so the circular-shift test moves samples that pass one end to the other. Whole-record conservation holds even when a wavelet spreads exposure over neighboring times. Checking how accurately the timeline locates a brief event is a separate test. A time–frequency sensitivity map would also need the Mark VII response to a change in each band's sound exposure.
