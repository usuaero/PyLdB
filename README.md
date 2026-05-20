# PyLdB

PyLdB calculates perceived loudness for pressure signatures using Stevens'
Mark VII perceived loudness method. It is intended for pressure histories such
as sonic boom signatures, with time in milliseconds and pressure in pounds per
square foot (psf).

The main interface is the `PyLdB` class. A calculator can ingest arrays directly
or read a two-column signature file compatible with `numpy.genfromtxt`. PANAIR
`.sig` files such as `misc/panair_r1.sig` can be loaded by skipping the three
header lines.

## Quick Start

```python
from pyldb import PyLdB

calculator = PyLdB("misc/panair_r1.sig", header_lines=3)
calculator.calculate(pad_front=6, pad_rear=6, len_window=800)

print(calculator.loudness.pldb)
print(calculator.loudness.dnl)
print(calculator.loudness.a_weighted_loudness)
```

After `calculate()` runs, results are available from `calculator.loudness`:

```python
calculator.loudness.pldb
calculator.loudness.ctl
calculator.loudness.ctl_response
calculator.loudness.community_tolerance_level
calculator.loudness.dnl
calculator.loudness.equivalent_loudness
calculator.loudness.a_weighted_loudness
calculator.loudness.b_weighted_loudness
calculator.loudness.c_weighted_loudness
calculator.loudness.d_weighted_loudness
calculator.loudness.alphabet_weighted_loudnesses
```

## Loading Data

Use `PyLdB` with a file:

```python
from pyldb import PyLdB

calculator = PyLdB("signature.sig", header_lines=3)
results = calculator.calculate()

print(results.pldb)
```

Or pass arrays directly:

```python
from pyldb import PyLdB

calculator = PyLdB(time=time_ms, pressure=pressure_psf)
results = calculator.calculate(pad_front=10, pad_rear=10)
```

The input arrays must be one-dimensional, equal length, strictly increasing in
time, and evenly sampled.

## Acoustic Metrics

`calculate()` computes PLdB first, then uses the acoustic helper methods to
populate additional metrics:

- `pldb`: Stevens Mark VII perceived loudness in PLdB.
- `ctl`: Fidell CTL high-annoyance response probability for the supplied
  community tolerance level.
- `community_tolerance_level`: CTL value used for the response calculation.
- `dnl`: day-night average level. If no day or night levels are supplied, PLdB
  is used as the representative level for all 24 hours.
- `equivalent_loudness`: energy-equivalent level. If no sequence is supplied,
  PLdB is used.
- `alphabet_weighted_loudnesses`: A-, B-, C-, and D-weighted overall levels
  calculated from the one-third-octave band sound pressure levels.

Optional inputs:

```python
results = calculator.calculate(
    ctl=75.0,
    growth=0.3,
    day_loudness=[70.0] * 15,
    night_loudness=[60.0] * 9,
    equivalent_loudnesses=[72.0, 75.0, 71.0],
)
```

## Legacy API

Function-style calls are still available for compatibility:

```python
import pyldb

time, pressure = pyldb.import_sig("misc/panair_r1.sig", header_lines=3)
pldb = pyldb.perceivedloudness(time, pressure, pad_front=6, pad_rear=6)
```

New code should prefer the `PyLdB` class because it exposes the full result set
through `calculator.loudness`.

## Installation

Clone the repository and install it in your Python environment:

```bash
git clone https://github.com/usuaero/PyLdB.git
cd PyLdB
pip install -e .
```

## Testing

Unit tests are implemented with `pytest`:

```bash
pytest verif/test.py
```

## Notes

PyLdB was supported by the NASA University Leadership Initiative (ULI) program
under federal award number NNX17AJ96A, titled Adaptive Aerostructures for
Revolutionary Civil Supersonic Transportation.

## Support

Contact doug.hunsaker@usu.edu or christianrbolander@gmail.com with questions.

## License

This project is licensed under the MIT license. See `LICENSE` for details.
