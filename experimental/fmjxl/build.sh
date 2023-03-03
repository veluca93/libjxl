#!/usr/bin/env bash
# Copyright (c) the JPEG XL Project Authors. All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.
set -e

DIR=$(realpath "$(dirname "$0")")
mkdir -p "$DIR"/build
cd "$DIR"/build

# set CXX to aarch64-linux-gnu-g++ if not set in the environment
CXX="${CXX-aarch64-linux-gnu-g++}"
if ! command -v "$CXX" >/dev/null ; then
  printf >&2 '%s: C++ compiler not found\n' "${0##*/}"
  exit 1
fi

"$CXX" -O3 -static -Wall \
  -I. \
  "$DIR"/fmjxl.cc "$DIR"/fmjxl_main.cc \
  -o fmjxl 
