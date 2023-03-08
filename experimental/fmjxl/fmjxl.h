#ifndef FMJXL_H
#define FMJXL_H

#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

struct FastMJXLEncoder;

// TODO(veluca): thread functions.
// width and height must be multiples of 16 and at least 256.
struct FastMJXLEncoder* FastMJXLCreateEncoder(
    size_t width, size_t height,
    void (*run_on_threads)(void*, void (*task)(void*, size_t shard), void*,
                           size_t count),
    void* thread_runner_data);

// Calling this function will clear the output buffer in the encoder of any of
// its current contents.
// It is invalid to call this function after a call with `is_last = 1`.
void FastMJXLAddYCbCrP010Frame(const uint8_t* y_plane, const uint8_t* uv_plane,
                               int is_last, struct FastMJXLEncoder* encoder);

const uint8_t* FastMJXLGetOutputBuffer(const struct FastMJXLEncoder* encoder);
uint8_t* FastMJXLReleaseOutputBuffer(struct FastMJXLEncoder* encoder);

// Returns the number of ready output bytes.
size_t FastMJXLGetOutputBufferSize(const struct FastMJXLEncoder* encoder);

void FastMJXLDestroyEncoder(struct FastMJXLEncoder* encoder);

#ifdef __cplusplus
}
#endif

#endif
