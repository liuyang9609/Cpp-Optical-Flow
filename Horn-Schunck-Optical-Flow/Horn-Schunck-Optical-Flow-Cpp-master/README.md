# Horn-Schunck-Optical-Flow-Cpp

C++ implementation of Horn-Schunck Optical Flow Algortithm

## Compile

The Horn-Schunck Optical Flow implementation is based on [CMake](http://www.cmake.org/) and OpenCV (for example, follow [Installing OpenCV in Xcode on macOS](https://medium.com/@jaskaranvirdi/setting-up-opencv-and-c-development-environment-in-xcode-b6027728003#:~:text=Install%20Xcode%20from%20the%20App,depending%20on%20the%20internet%20speed.&text=This%20should%20install%20OpenCV%203.). The code has been tested on macOS Catalina 10.15.5 and Apple clang version 11.0.3.

    $ cd Horn-Schunck-Optical-Flow-Cpp
    $ mkdir build
    $ cd build
    $ cmake ..
    $ make -j4

## Usage

This implementation can be run on two types of input: images or video. Based on the mode of input, first three arguments will change.

    ./hscpp ['image' (or) mp4 input path] [prev image path (or) prev frame number] [next image path (or) next frame number] [save path for results] [hs (or) fb]

