// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This program decodes a JPEG XL file to RGBA buffers, using a custom thread
// pool.

#include <stdio.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <fstream>
#include <mutex>
#include <thread>
#include <vector>

#include "jxl/decode.h"
#include "jxl/decode_cxx.h"
#include "jxl/thread_parallel_runner.h"
#include "jxl/thread_parallel_runner_cxx.h"

// A thread pool that allows changing the number of threads it runs. It also
// runs tasks on the calling thread, which can work better on schedulers for
// heterogeneous architectures.
struct ResizeableParallelRunner {
  void SetNumThreads(size_t num) {
    if (num > 0) {
      num -= 1;
    }
    {
      std::unique_lock<std::mutex> l(m_);
      num_desired_workers_ = num;
      cv_.notify_all();
    }
    if (workers_.size() < num) {
      for (size_t i = workers_.size(); i < num; i++) {
        workers_.emplace_back([this, i]() { WorkerBody(i); });
      }
    }
    if (workers_.size() > num) {
      for (size_t i = num; i < workers_.size(); i++) {
        workers_[i].join();
      }
      workers_.resize(num);
    }
  }

  ~ResizeableParallelRunner() { SetNumThreads(0); }

  JxlParallelRetCode Run(void *jxl_opaque, JxlParallelRunInit init,
                         JxlParallelRunFunction func, uint32_t start,
                         uint32_t end) {
    if (start + 1 == end) {
      JxlParallelRetCode ret = init(jxl_opaque, 1);
      if (ret != 0) return ret;

      func(jxl_opaque, start, 0);
      return ret;
    }

    size_t num_workers = std::min<size_t>(workers_.size() + 1, end - start);
    JxlParallelRetCode ret = init(jxl_opaque, num_workers);
    if (ret != 0) {
      return ret;
    }

    {
      std::unique_lock<std::mutex> l(m_);
      // Avoid waking up more workers than needed.
      max_running_workers_ = end - start - 1;
      next_task_ = start;
      end_task_ = end;
      func_ = func;
      jxl_opaque_ = jxl_opaque;
      work_available_ = true;
      cv_.notify_all();
    }

    DequeueTasks(0);

    while (true) {
      std::unique_lock<std::mutex> l(m_);
      if (num_running_workers_ == 0) break;
      cv_.wait(l);
    }

    return ret;
  }

  static JxlParallelRetCode Run(void *runner_opaque, void *jxl_opaque,
                                JxlParallelRunInit init,
                                JxlParallelRunFunction func, uint32_t start,
                                uint32_t end) {
    return static_cast<ResizeableParallelRunner *>(runner_opaque)
        ->Run(jxl_opaque, init, func, start, end);
  }

 private:
  void WorkerBody(size_t worker_id) {
    while (true) {
      {
        std::unique_lock<std::mutex> l(m_);
        // Worker pool was reduced, resize down.
        if (worker_id >= num_desired_workers_) {
          return;
        }
        // Nothing to do this time.
        if (!work_available_ || worker_id >= max_running_workers_) {
          cv_.wait(l);
          continue;
        }
      }
      DequeueTasks(worker_id + 1);
    }
  }

  void DequeueTasks(size_t thread_id) {
    {
      std::unique_lock<std::mutex> l(m_);
      num_running_workers_++;
    }
    while (true) {
      uint32_t task = next_task_++;
      if (task >= end_task_) {
        std::unique_lock<std::mutex> l(m_);
        num_running_workers_--;
        work_available_ = false;
        if (num_running_workers_ == 0) {
          cv_.notify_all();
        }
        break;
      }
      func_(jxl_opaque_, task, thread_id);
    }
  }

  std::atomic<uint32_t> next_task_;
  uint32_t end_task_;
  JxlParallelRunFunction func_;
  void *jxl_opaque_;  // not owned

  std::mutex m_;
  std::condition_variable cv_;
  size_t num_desired_workers_ = 0;
  size_t max_running_workers_ = 0;
  size_t num_running_workers_ = 0;
  bool work_available_ = false;

  std::vector<std::thread> workers_;
};

bool DecodeJpegXlOneShot(const uint8_t *jxl, size_t size,
                         std::vector<uint8_t> *pixels, size_t *xsize,
                         size_t *ysize, std::vector<uint8_t> *icc_profile,
                         size_t num_threads) {
  ResizeableParallelRunner runner;
  runner.SetNumThreads(num_threads);

  auto dec = JxlDecoderMake(nullptr);
  if (JXL_DEC_SUCCESS !=
      JxlDecoderSubscribeEvents(dec.get(), JXL_DEC_BASIC_INFO |
                                               JXL_DEC_COLOR_ENCODING |
                                               JXL_DEC_FULL_IMAGE)) {
    fprintf(stderr, "JxlDecoderSubscribeEvents failed\n");
    return false;
  }

  if (JXL_DEC_SUCCESS !=
      JxlDecoderSetParallelRunner(dec.get(), ResizeableParallelRunner::Run,
                                  &runner)) {
    fprintf(stderr, "JxlDecoderSetParallelRunner failed\n");
    return false;
  }

  JxlBasicInfo info;
  JxlPixelFormat format = {3, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0};

  JxlDecoderSetInput(dec.get(), jxl, size);

  for (;;) {
    switch (JxlDecoderProcessInput(dec.get())) {
      case JXL_DEC_ERROR:
        fprintf(stderr, "Decoder error\n");
        return false;
      case JXL_DEC_NEED_MORE_INPUT:
        fprintf(stderr, "Error, already provided all input\n");
        return false;
      case JXL_DEC_BASIC_INFO:
        if (JXL_DEC_SUCCESS != JxlDecoderGetBasicInfo(dec.get(), &info)) {
          fprintf(stderr, "JxlDecoderGetBasicInfo failed\n");
          return false;
        }
        *xsize = info.xsize;
        *ysize = info.ysize;
        break;
      case JXL_DEC_COLOR_ENCODING: {
        size_t icc_size;
        if (JXL_DEC_SUCCESS !=
            JxlDecoderGetICCProfileSize(
                dec.get(), &format, JXL_COLOR_PROFILE_TARGET_DATA, &icc_size)) {
          fprintf(stderr, "JxlDecoderGetICCProfileSize failed\n");
          return false;
        }
        icc_profile->resize(icc_size);
        if (JXL_DEC_SUCCESS != JxlDecoderGetColorAsICCProfile(
                                   dec.get(), &format,
                                   JXL_COLOR_PROFILE_TARGET_DATA,
                                   icc_profile->data(), icc_profile->size())) {
          fprintf(stderr, "JxlDecoderGetColorAsICCProfile failed\n");
          return false;
        }
        break;
      }
      case JXL_DEC_NEED_IMAGE_OUT_BUFFER: {
        size_t buffer_size;
        if (JXL_DEC_SUCCESS !=
            JxlDecoderImageOutBufferSize(dec.get(), &format, &buffer_size)) {
          fprintf(stderr, "JxlDecoderImageOutBufferSize failed\n");
          return false;
        }
        if (buffer_size != *xsize * *ysize * 3) {
          fprintf(stderr, "Invalid out buffer size %zu %zu\n", buffer_size,
                  *xsize * *ysize * 3);
          return false;
        }
        pixels->resize(*xsize * *ysize * 3);
        void *pixels_buffer = (void *)pixels->data();
        size_t pixels_buffer_size = *xsize * *ysize * 3;
        if (JXL_DEC_SUCCESS !=
            JxlDecoderSetImageOutBuffer(dec.get(), &format, pixels_buffer,
                                        pixels_buffer_size)) {
          fprintf(stderr, "JxlDecoderSetImageOutBuffer failed\n");
          return false;
        }
        break;
      }
      case JXL_DEC_FULL_IMAGE:
        break;
      case JXL_DEC_SUCCESS:
        return true;
      default:
        fprintf(stderr, "Unknown decoder status\n");
        return false;
    }
  }
}

bool LoadFile(const char *filename, std::vector<uint8_t> *out) {
  std::ifstream file(filename, std::ios::in | std::ios::binary);
  if (!file) return false;
  if (!file.seekg(0, std::ios::end)) return false;
  out->resize(file.tellg());
  if (!file.seekg(0)) return false;
  if (!file.read(reinterpret_cast<char *>(out->data()), out->size()))
    return false;
  file.close();
  if (!file) return false;
  return true;
}

int main(int argc, char *argv[]) {
  if (argc < 4) {
    fprintf(stderr,
            "Usage: %s <jxl> <num_threads> <num_reps>\n"
            "Where:\n"
            "  jxl = input JPEG XL image filename\n"
            "  num_threads = Number of threads, 0 for “auto”\n"
            "  num_reps = Number of repetitions\n",
            argv[0]);
    return 1;
  }

  const char *out_filename = argc >= 5 ? argv[4] : nullptr;

  const char *jxl_filename = argv[1];
  const char *num_threads_str = argv[2];
  const char *num_reps_str = argv[3];

  char *end;
  const size_t num_threads = strtol(num_threads_str, &end, 10);
  if (*num_threads_str == '\0' || *end != '\0') {
    fprintf(stderr, "Failed to parse “%s” as a number of threads.",
            num_threads_str);
    return 1;
  }
  const size_t num_reps = strtol(num_reps_str, &end, 10);
  if (*num_reps_str == '\0' || *end != '\0') {
    fprintf(stderr, "Failed to parse “%s” as a number of repetitions.",
            num_reps_str);
    return 1;
  }

  std::vector<uint8_t> jxl;
  if (!LoadFile(jxl_filename, &jxl)) {
    fprintf(stderr, "couldn't load %s\n", jxl_filename);
    return 1;
  }

  double log_geomean = 0;
  double min_speed = std::numeric_limits<double>::infinity();
  double max_speed = -std::numeric_limits<double>::infinity();
  // Perform one more iteration at the beginning so that we can discard it.
  for (size_t i = 0; i < 1 + num_reps; ++i) {
    const auto start = std::chrono::steady_clock::now();
    std::vector<uint8_t> pixels;
    std::vector<uint8_t> icc_profile;
    size_t xsize = 0, ysize = 0;
    if (!DecodeJpegXlOneShot(jxl.data(), jxl.size(), &pixels, &xsize, &ysize,
                             &icc_profile, num_threads)) {
      fprintf(stderr, "Error while decoding the jxl file\n");
      return 1;
    }
    const auto end = std::chrono::steady_clock::now();

    const std::chrono::duration<double, std::ratio<1>> elapsed = end - start;
    const double speed = xsize * ysize * 1e-6 / elapsed.count();

    if (i == 0) {
      if (out_filename) {
        FILE *out = fopen(out_filename, "w");
        fprintf(out, "P6\n%lu %lu\n255\n", xsize, ysize);
        fwrite(pixels.data(), 1, pixels.size(), out);
        fclose(out);
      }
      continue;
    }

    if (speed < min_speed) min_speed = speed;
    if (speed > max_speed) max_speed = speed;
    log_geomean += std::log(speed);
  }

  log_geomean /= num_reps;
  const double geomean = std::exp(log_geomean);

  printf("geomean: %g MP/s [%g, %g]\n", geomean, min_speed, max_speed);

  return 0;
}
