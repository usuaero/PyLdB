# PyLdB

This package provides a perceived loudness calculator implementing Stevens' perceived loudness methodology (see
References) for predicting the PLdB (perceived loudness in decibels) of a pressure signature. PyLdB was created
in an effort to make this functionality more readily available to scientists and engineers, as the other option
available requires following ITAR training and procedures to use. 

PyLdB requires both time and pressure arrays in milliseconds and pounds per square foot respectively as an input.
It also is able to import these arrays from a file, as long as the file is compatible with the numpy genfromtxt
function.

The following code demonstrates how PyLdB can be used in a Python script. Specifically, it will demonstrate how
to import time and pressure arrays from a file, and then use those arrays to calculate the PLdB of the pressure
signature.

```python
import pyldb

time, pressure = pyldb.import_sig("pyldb_sig1.sig", header_lines=3)
PLdB = pyldb.perceivedloudness(time, pressure, front_pad=10, rear_pad=10)

print(PLdB)
```

## Notes

PyLdB was supported by the NASA University Leadership Initiative (ULI) program under federal award number
NNX17AJ96A, titled Adaptive Aerostructures for Revolutionary Civil Supersonic Transportation.

## Documentation

See doc strings in code.

## Installation

You can either download the source as a ZIP file and extract the contents or clone the pyldb repository using Git.

### Downloading source as a ZIP file

1. Open a web browser and navigate to [https://github.com/usuaero/pyldb](https://github.com/usuaero/pyldb)
2. Make sure the branch is set to 'Master'
3. Click the `Clone or download` button
4. Select `Download ZIP`
5. Extract the downloaded ZIP file to a local directory on your machine

###Cloning the Github repository

1. From the command prompt, navigate to the directory where pyldb will be installed
2. `git clone https://github.com/usuaero/pyldb`

## Testing

Unit tests are implemented using the pytest module and are run using the following command.

`python3 -m pytest test/`

## Support

Contact doug.hunsaker@usu.edu with any questions.

## License

This project is licensed under the MIT license. See LICENSE file for more information.      
