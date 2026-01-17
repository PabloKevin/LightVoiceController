#include "feature_provider.h"
#include "audio_config.h"
#include <cstring>

static float mfcc_buffer[MFCC_FRAMES][MFCC_COEFFS];
static int frame_index = 0;

bool feature_add_frame(const float* mfcc_frame) {
    memcpy(mfcc_buffer[frame_index], mfcc_frame,
           MFCC_COEFFS * sizeof(float));
    frame_index++;

    if (frame_index >= MFCC_FRAMES) {
        return true;
    }
    return false;
}

void feature_get(float* out) {
    memcpy(out, mfcc_buffer,
           MFCC_FRAMES * MFCC_COEFFS * sizeof(float));
}

void feature_reset() {
    frame_index = 0;
}
