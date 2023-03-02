#!/usr/bin/env bash
# Copyright (c) the JPEG XL Project Authors. All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.
set -e

DIR=$(realpath "$(dirname "$0")")

mkdir -p /tmp/build-android
cd /tmp/build-android

CXX="$ANDROID_NDK"/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android30-clang++
if ! command -v "$CXX" >/dev/null ; then
  printf >&2 '%s: Android C++ compiler not found, is ANDROID_NDK set properly?\n' "${0##*/}"
  exit 1
fi

"$CXX" -O3 \
  -I"${DIR}" \
  "${DIR}"/fmjxl.cc "${DIR}"/fmjxl_main.cc \
  -o fmjxl
