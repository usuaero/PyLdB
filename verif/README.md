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

The wavelet procedure is evaluated by comparing the sound exposure in each one-third-octave band and the resulting perceived level (PLdB) with the values calculated by PyLdB 1.0.0 for the same pressure signature. In both calculations, band sound exposure is the time integral of squared band pressure and has units of Pa²·s. The band exposure timeline gives the exposure rate, in Pa², at each sample. Sound exposure level expresses exposure relative to a reference in decibels. [ASA/ANSI sound exposure](https://asastandards.org/terms/sound-exposure/) · [ASA/ANSI sound exposure level](https://asastandards.org/terms/sound-exposure-level/)

### Calculation procedure

The calculation begins with the FFT of the tapered and zero-padded pressure signature. Each FFT bin represents a finite frequency interval. Let $\alpha_{jk}$ be the fraction of bin $k$ that overlaps band $j$. The complex coefficient assigned to the band is the original coefficient multiplied by $\sqrt{\alpha_{jk}}$. For example, a bin with 60% overlap contributes 60% of its exposure because scaling its amplitude by $\sqrt{0.60}$ scales its squared magnitude by 0.60. The phase of the coefficient is preserved. [NASA TM-20250003346, Appendix 1](https://ntrs.nasa.gov/citations/20250003346)

Six Morlet frequency responses are evaluated for each band. The wavelet scale determines its duration and center frequency, and the six centers are spaced logarithmically within the band. Let $g_{mk}$ denote the gain, or amplitude response, of wavelet $m$ at frequency bin $k$. Each column of the gain matrix is normalized by dividing its gains by $\sqrt{\sum_m |g_{mk}|^2}$. The sum of the squared normalized gains is then unity at each frequency, and each squared gain gives the fraction of that bin's assigned exposure carried by the corresponding wavelet. The normalization depends on the wavelet responses and the FFT frequency grid. The whole-record FFT exposure totals are used only for comparison.

For each wavelet, the complex band coefficients are multiplied by the corresponding row of normalized gains and transformed back to time using a complex inverse FFT. The resulting wavelet coefficients describe the response to the pressure signature at each sample time. For a band with $M$ participating positive-frequency FFT bins and a record of $N$ samples, the gain matrix has dimensions $6 \times M$ and the wavelet coefficient matrix has dimensions $6 \times N$.

The inverse transforms use just the one-sided spectrum. Multiplying positive-frequency bins with a negative-frequency partner by $\sqrt{2}$ restores their two-sided exposure. The even-length Nyquist bin has no separate partner and is counted once. The Morlet wavelets have zero gain at zero frequency, so any assigned exposure from the zero-frequency bin is carried as a constant rate over the record.

The exposure rate $q_j[n]$ at sample $n$ is obtained by summing the squared magnitudes of the six wavelet coefficients for band $j$ and including any contribution from the zero-frequency bin. The exposure over the record is

$$
E_j = \Delta t \sum_{n=0}^{N-1} q_j[n],
$$

where $\Delta t$ is the sample interval. By Parseval's identity, the time-domain sum of squares and the corresponding frequency-domain sum agree when the FFT scaling is included. Since the squared gains sum to unity at each bin, $E_j$ equals the exposure assigned to band $j$ in exact arithmetic. The recovered band exposures are used to calculate PLdB with the same 70 ms normalization, −3 dB band adjustment, and Mark VII procedure as PyLdB 1.0.0. [Bolander et al., 2019, Eqs. (6)–(10)](https://doi.org/10.2514/6.2019-2091)

### Two-sinusoid example

Consider a one-second pressure signature consisting of 96 Hz and 104 Hz sinusoids, each with a peak amplitude of 1 Pa. The 104 Hz sinusoid has a 60° phase offset. Both FFT bins fall within the nominal 100 Hz band. The exposure of each sinusoid is 0.5 Pa²·s, which gives a total band exposure of 1.0 Pa²·s.

A beat is the periodic increase and decrease in the combined response as two sinusoids with nearby frequencies alternately reinforce and partly cancel. The frequency difference in this case is 8 Hz, which gives a period of 125 ms in the exposure timeline. The calculated exposure rate is compared at each sample with the analytical beat, including its phase and amplitude. The time-average rate is 1 Pa², which gives an exposure of 1.0 Pa²·s over the record.

### Results

The wavelet and baseline FFT calculations agree to eight decimal places in PLdB for all eight verification waveforms. The maximum absolute difference in band sound exposure over these cases was 7.105 × 10⁻¹⁵ Pa²·s.

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

The analytical cases also verify the treatment of zero pressure, zero frequency, and the highest bins of odd- and even-length records. The Fourier calculation treats the processed signature as periodic. In the circular-shift test, samples moved past one end of the record reenter at the other, and the exposure timeline shifts by the same number of samples. These results establish full-record exposure conservation and verify the time dependence of the periodic example. The accuracy of exposure localization for a brief event depends on the wavelet duration and requires a separate assessment. A time–frequency sensitivity assessment also requires the change in Mark VII perceived level with respect to band sound exposure.
