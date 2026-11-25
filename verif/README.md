# Stevens Mark VII verification

This compares the PyLdB calculation at `64264fe` (the merge of PR #4) with an independent implementation of the published method. The existing 77.67985 PLdB PANAIR result remains a regression check. The four LCASB comparisons in the 2019 PyLdB paper used other signatures. This comparison uses the NASA equations directly and does not run LCASB, NOM, or PCBoom. [Bolander et al., 2019, Table 3](https://doi.org/10.2514/6.2019-2091)

## Reproduce

With PyLdB and pytest installed, run from the repository root:

```sh
pytest --import-mode=importlib -o pythonpath=verif verif/test.py verif/test_mark_vii.py
python verif/compare_mark_vii.py
```

The comparison prints the energy, SPL, equivalent level, and sones in the loudest bands, plus total sones and the effect of each calculation stage on PLdB. The tests check the published 3150 Hz reference, the two-band summation, three quiet spectra from NASA Table 2, and the exact low-level summation breakpoints. Analytical checks cover FFT energy conservation, band-edge allocation, the Nyquist limit, and the Fourier spectrum of the NASA N-wave. The Table 2 test allows 0.003 PLdB because rounding the published low-frequency SPLs is amplified by the steep contours and by the summation factor near 0.181 sone. [NASA TM-20250003346, Table 2](https://ntrs.nasa.gov/citations/20250003346)

## Method and inputs

- The reference computes one-sided bin exposure in Pa² s from the true sample spacing and allocates it by fractional overlap with base-10 one-third-octave bands. It normalizes band exposure over 70 ms with a 20 µPa reference pressure. PyLdB derives FFT spacing from the endpoint span divided by the sample count, interpolates to tabulated band edges, and integrates by trapezoids. The band-energy column in the comparison converts PyLdB's stored mean-square pressure back to exposure. [NASA TM-20250003346, Appendix 1](https://ntrs.nasa.gov/citations/20250003346) · [PCBoom Technical Reference, 2nd ed.](https://ntrs.nasa.gov/citations/20250003228)
- The reference uses the Jackson–Leventhall band-number equations from Table 2 and Appendix A of the 2019 paper. It uses the limiting contour at 1 Hz and has exact frequency centers. PyLdB uses the tabulated nominal center frequencies and a near-1-Hz proxy in its logarithmic formula. [Bolander et al., 2019](https://doi.org/10.2514/6.2019-2091)
- The reference uses Klos's piecewise level-to-sone and sone-to-level equations. It weights the sum of the bands other than the loudest by Stevens' interpolated factor, with NASA's two added quiet-level points. PyLdB interpolates a rounded equivalent-level-to-sone table and sets the summation factor to zero below 0.181 sone. Its final logarithm yields `-inf` at zero sones, where the reference uses the −3 dB lower limit. [Stevens, 1972, Appendix B](https://doi.org/10.1121/1.1912880) · [NASA TM-20250003346, Sections II.C–D](https://ntrs.nasa.gov/citations/20250003346)
- The PANAIR input and its 1%, 0.4%, and 0.1% pressure-scaled variants use an 800-sample Hann taper at each end and six record lengths of zero padding on each side. The 48 kHz NASA N-wave has 50 Pa peak overpressure, 8 ms rise time, and 350 ms duration. It starts and ends at zero and uses three record lengths of padding on each side. The untapered periodic probes use 1 Pa peak tones at 3162 Hz and 89 Hz in one-second, 32 kHz records. The 8 kHz case combines a 1 Pa peak 500 Hz cosine with a 0.01 Pa Nyquist component over 512 samples. [PANAIR fixture](../misc/panair_r1.sig) · [NASA TP-3134, Figures 1–2 and Eq. (1)](https://ntrs.nasa.gov/citations/19920002547)
- Both calculations use the same processed samples from each waveform. The reference runs at −3 dB and 0 dB band adjustments so the event convention can be measured independently. PyLdB always subtracts 3 dB. NASA's PCBoom reference describes that adjustment for a two-event N-wave with shocks separated by more than 70 ms. The tone cases use both settings to isolate the numerical effect. [PCBoom Technical Reference, 2nd ed.](https://ntrs.nasa.gov/citations/20250003228)

## Results

The signed differences below are PyLdB minus the corresponding reference result. Values are in PLdB.

| Waveform | PyLdB | Reference with −3 dB | Difference | Reference with 0 dB | Difference |
|---|---:|---:|---:|---:|---:|
| PANAIR | 77.680 | 77.707 | −0.027 | 81.111 | −3.431 |
| PANAIR at 1% amplitude | 27.934 | 28.153 | −0.219 | 32.815 | −4.881 |
| PANAIR at 0.4% amplitude | 7.859 | 12.147 | −4.288 | 18.807 | −10.948 |
| PANAIR at 0.1% amplitude | −∞ | −3.000 | −∞ | −1.401 | −∞ |
| NASA N-wave | 89.741 | 89.810 | −0.068 | 93.254 | −3.513 |
| Reference-band tone | 99.517 | 99.518 | −0.001 | 102.518 | −3.001 |
| Band-edge tone | 80.987 | 81.124 | −0.137 | 84.145 | −3.157 |
| 8 kHz sampling | 84.725 | 79.843 | +4.882 | 82.845 | +1.879 |

Replacing reference stages sequentially with PyLdB values assigns these PLdB increments. This attribution depends on the order because the downstream calculation is nonlinear.

| Waveform | Band SPL | Contour | Band sones | Summation | Final conversion |
|---|---:|---:|---:|---:|---:|
| PANAIR | +0.001 | −0.033 | +0.005 | 0.000 | 0.000 |
| PANAIR at 1% amplitude | −0.008 | −0.153 | −0.071 | 0.000 | +0.012 |
| PANAIR at 0.4% amplitude | −0.028 | −0.293 | −0.617 | −3.086 | −0.265 |
| PANAIR at 0.1% amplitude | 0.000 | 0.000 | 0.000 | 0.000 | −∞ |
| NASA N-wave | +0.005 | −0.079 | +0.006 | 0.000 | 0.000 |
| Reference-band tone | −0.00014 | 0.000 | −0.00132 | 0.000 | 0.000 |
| Band-edge tone | −0.140 | 0.000 | +0.004 | 0.000 | 0.000 |
| 8 kHz sampling | +4.874 | 0.000 | +0.008 | 0.000 | 0.000 |

The contour stage accounts for most of the difference on the unscaled PANAIR and N-wave cases. In the 8 kHz case, PyLdB assigns 0.00393 Pa² s to bands entirely above Nyquist because the frequency interpolation extends the endpoint level. The band SPL stage also explains most of the 89 Hz edge-tone discrepancy. A 31-sample cosine at its highest positive FFT bin recovered only 48.4% of time-domain energy from PyLdB's raw FFT, measured by summing the spectrum before band-edge interpolation and multiplying by bin width. The even 32-sample Nyquist case recovered 96.9%, consistent with the endpoint-based sample interval alone. The 0.4%-amplitude case exercises the missing quiet summation points. Both methods reach zero sones in the 0.1%-amplitude case with the matched correction, exposing the final conversion difference.
