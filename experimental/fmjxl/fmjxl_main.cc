// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "fmjxl.h"

struct SimpleThreadPool {
  std::vector<std::thread> threads;
  std::mutex task_m;
  std::condition_variable cv;
  void (*task)(void*, size_t);
  void* opaque;
  size_t shard_count;
  size_t running_tasks;
  std::atomic<size_t> current_shard;

  explicit SimpleThreadPool(size_t num) {
    shard_count = 1;
    task = nullptr;
    running_tasks = 0;
    for (size_t i = 0; i < num - 1; i++) {
      threads.emplace_back([&]() { this->ThreadFn(); });
    }
  }

  ~SimpleThreadPool() {
    {
      std::unique_lock<std::mutex> lock(task_m);
      shard_count = 0;
      task = nullptr;
      cv.notify_all();
    }
    for (std::thread& t : threads) t.join();
  }

  void ThreadFn() {
    while (true) {
      {
        // Wait for a task to become available.
        std::unique_lock<std::mutex> lock(task_m);
        cv.wait(lock);
        if (task == nullptr) {
          if (shard_count == 0) return;
          // Spurious wakeup.
          continue;
        }
        if (current_shard >= shard_count) {
          continue;
        }
        running_tasks += 1;
      }

      // Run enqueued tasks.
      while (true) {
        size_t shard = current_shard++;
        if (shard >= shard_count) break;
        task(opaque, shard);
      }

      // Done running all shards, signal we are not running anymore.
      {
        std::unique_lock<std::mutex> lock(task_m);
        running_tasks -= 1;
        cv.notify_all();
      }
    }
  }

  void Run(void (*t)(void*, size_t), void* o, size_t n) {
    {
      std::unique_lock<std::mutex> lock(task_m);
      task = t;
      opaque = o;
      shard_count = n;
      current_shard = 0;
      cv.notify_all();
    }

    // Main thread participates in running tasks.
    while (true) {
      size_t shard = current_shard++;
      if (shard >= shard_count) break;
      task(opaque, shard);
    }

    // Wait for all currently-running tasks to finish.
    while (true) {
      std::unique_lock<std::mutex> lock(task_m);
      if (running_tasks == 0) {
        break;
      }
      cv.wait(lock);
    }
  }
};

int main(int argc, char** argv) {
  if (argc < 7) {
    fprintf(stderr,
            "Usage: %s out.jxl w h num_reps num_threads qtype f1_ycbcr10.bin "
            "[f2_ycbcr10.bin [...]] \n",
            argv[0]);
    return 1;
  }
  size_t w = atoi(argv[2]);
  size_t h = atoi(argv[3]);
  assert(w % 16 == 0);
  assert(h % 16 == 0);
  assert(w > 256 || h > 256);
  size_t num_reps = atoi(argv[4]);
  size_t num_threads = atoi(argv[5]);
  QuantizationType qtype = (QuantizationType)atoi(argv[6]);
  assert(num_threads > 0);

  SimpleThreadPool thread_pool(num_threads);

  assert(num_reps > 0);

  std::vector<std::vector<uint8_t>> input_data;

  for (size_t f = 0; f + 7 < (size_t)argc; f++) {
    FILE* in = fopen(argv[7 + f], "r");
    assert(in);
    fseek(in, 0, SEEK_END);
    ssize_t size = ftell(in);
    fseek(in, 0, SEEK_SET);
    input_data.emplace_back(size, 0);
    fread(input_data.back().data(), 1, size, in);
    fclose(in);
  }

  std::vector<std::pair<std::unique_ptr<uint8_t[], void (*)(void*)>, size_t>>
      encoded_chunks;
  size_t encoded_size = 0;

  auto start = std::chrono::high_resolution_clock::now();
  for (size_t _ = 0; _ < num_reps; _++) {
    encoded_chunks.clear();
    encoded_size = 0;
    auto encoder = FastMJXLCreateEncoder(
        w, h,
        +[](void* runner, void (*tfn)(void*, size_t), void* opaque, size_t n) {
          static_cast<SimpleThreadPool*>(runner)->Run(tfn, opaque, n);
        },
        &thread_pool);
    for (size_t i = 0; i < input_data.size(); i++) {
      FastMJXLAddYCbCrP010Frame(input_data[i].data(),
                                input_data[i].data() + 2 * w * h, qtype,
                                i + 1 == input_data.size(), encoder);
      size_t chunk_size = FastMJXLGetOutputBufferSize(encoder);
      std::unique_ptr<uint8_t[], decltype(&free)> chunk{
          FastMJXLReleaseOutputBuffer(encoder), free};
      encoded_size += chunk_size;
      encoded_chunks.emplace_back(std::move(chunk), chunk_size);
    }
    FastMJXLDestroyEncoder(encoder);
  }
  auto stop = std::chrono::high_resolution_clock::now();

  if (num_reps > 1) {
    float us =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
            .count();
    size_t pixels = w * h * num_reps * input_data.size();
    float mps = pixels / us;
    fprintf(stderr, "%10.3f MP/s\n", mps);
    fprintf(
        stderr, "%10.3f bits/pixel\n",
        encoded_size * 8.0 / (float(w) * float(h) * float(input_data.size())));
  }

  FILE* out = fopen(argv[1], "w");
  assert(out);
  for (const auto& v : encoded_chunks) {
    fwrite(v.first.get(), v.second, 1, out);
  }
  fclose(out);
}
