
# CVzero
### Making computer vision simple for everyone!

Not much going on with this at the moment :-(

PLEASE NOTE: It still early days and we don't have running code or example code, watch this space!

CVzero is a Python 3 library that aims to simplify computer vision for the Raspberry Pi* robotics.
As well as the library, a companion App for the Raspberry Pi is being developed for selecting an object and displaying its colour signature. It will be possible to save the colour signatures collected as a file for the library to use.

Please be patient with us as we are all volunteers and have a life outside of OpenCV

The CVzero library requires that OpenCV and its dependencies are installed.

We have a Discord channel for chat and collaboration, https://discord.gg/Vbdyjdn. Join with this URL!

*The library may be able to be used with other platforms but has not been tested or guaranteed to work.

## Recommended hardware

### Raspberry Pi

Raspberry Pi 3 A, B and B+
Raspberry Pi 4 B

### Camera

Offical Raspberry Pi camera, v1 or v2
linux support Webcams

## Design document 

https://docs.google.com/document/d/1-CuN453Kmf-dKiIAXQaXqBjWy_byByh1wRkkukcPAqQ/edit?usp=sharing

## How to help

Join us on our discord channel, download and test the libray and example code.
If you wish to submit code please make a pull request.


## Building

This code builds using setuptools. We strongly suggest you use a virtual environment, and then use 
```python setup.py develop``` - this will symlink into your python installation such that any changes
you make to the code here will be immediately reflected in the libraries imported by scripts without
having to install again.




