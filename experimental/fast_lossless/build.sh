#!/bin/bash -e


mkdir -p build
cd build

[ -f lodepng.cpp ] || wget https://raw.githubusercontent.com/lvandeve/lodepng/8c6a9e30576f07bf470ad6f09458a2dcd7a6a84a/lodepng.cpp
[ -f lodepng.h ] || wget https://raw.githubusercontent.com/lvandeve/lodepng/8c6a9e30576f07bf470ad6f09458a2dcd7a6a84a/lodepng.h
[ -f lodepng.o ] || g++ lodepng.cpp -O3 -mavx2 -o lodepng.o -c

clang++ -O3 -mavx2 -g \
  -Ibuild/ -I. -I../../../ -I../../../third_party/highway/ \
  lodepng.o \
  ../fast_lossless.cc ../fast_lossless_main.cc \
  -o fast_lossless
