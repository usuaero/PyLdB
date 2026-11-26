# Changelog

## 1.0.0

### Fixed

- Use the actual sample interval and correct the one-sided FFT scaling for odd-length records.
- Allocate FFT bin energy by overlap with exact base-10 one-third-octave bands. Bands entirely above Nyquist receive zero energy. The saved spectrum contains the raw one-sided FFT grid.
- Use exact band numbers in the Jackson–Leventhall loudness contours, including the explicit 1 Hz limit.
- Use the published low-level sone conversions and NASA's two quiet-level additions to Stevens' summation table. Correct the summation-table entry from 1.71 to 1.72 sones. Zero total sones gives −3 PLdB.

### Output changes

With the same 70 ms normalization and −3 dB band adjustment, the checked-in PANAIR waveform changed from **77.67985** to **77.70673 PLdB**. The 8 kHz sampling probe changed from **84.72461** to **79.84261 PLdB**. PANAIR at 0.4% amplitude changed from **7.85900** to **12.14730 PLdB**, and at 0.1% amplitude from **−∞** to **−3.00000 PLdB**. [Current verification results](verif/README.md)
