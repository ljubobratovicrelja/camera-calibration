# Camera Calibration

**Bachelor of Engineering thesis, Faculty of Technical Science, Novi Sad, Serbia, 2015**


Implementation and revision of Zhengyou Zhang's algorithm for camera calibration, 
*Flexible Camera Calibration By Viewing a Plane From Unknown Orientations*, 1999.

[Here is compiled version for Windows 8, compiled using Visual Studio 2013.](https://www.dropbox.com/s/tot12zm0qi2rsl1/camera_calibration_vc12_x64.zip?dl=0)

All needed dlls should be there - if there's any missing, please let me know via email, it would help a lot!


## Program instructions
Program offers pattern detection routine, and calibration routine. Calibration 
pattern detection can be triggered using flags:

```
camera_calibration --detection --p-rows 6 --p-cols 9
```

After storing the calibration pattern corners (let's say you've stored it as **pattern.data**), path to that stored file needs to be given as first 
argument for calibration program:

```
camera_calibration ./path/to/pattern.data
```

Program should output the calibration results, and write reprojection points in the same directory as executable, as reprojection_*.png.

## Dependencies

This project relies on LibCV library, present in other repository of mine:
https://github.com/ljubobratovicrelja/libcv

It is developed using the version of the library which was tagged as v0.1
in github, and it can be downloaded as that *release* tag:
https://github.com/ljubobratovicrelja/libcv/releases/tag/v0.1


## Compilation

Compilation should be straight-forward on any linux distro using GCC with c++11 support, 
and is tested on Windows 8 using VC12. See LibCV readme for more info on compilation.
As long as the LibCV is compiled, compilation of this project should be easy using CMake (GUI).
 
