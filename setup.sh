#!/bin/sh
git submodule init
git submodule update

cd external/blaze
mkdir build
cd build
cmake .. -DBLAZE_SMP_THREADS=C++11
cd ../../..
