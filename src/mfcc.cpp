#include "mfcc.h"
#include "tensorflow/lite/experimental/microfrontend/lib/frontend.h"

static FrontendState g_state;
static FrontendConfig g_config;

void mfcc_init() {
    g_config.window.size_ms = 20;
    g_config.window.step_size_ms = 20;
    g_config.filterbank.num_channels = MFCC_COEFFS;
    g_config.filterbank.lower_band_limit = 20;
    g_config.filterbank.upper_band_limit = 4000;
    g_config.noise_reduction.enable_noise_reduction = true;
    g_config.pcan_gain_control.enable_pcan = true;
    g_config.log_scale.enable_log = true;

    FrontendPopulateState(&g_state, &g_config, SAMPLE_RATE);
}

void mfcc_compute(const int16_t* audio, float* mfcc_out) {
    FrontendOutput output = FrontendProcessSamples(
        &g_state,
        audio,
        FRAME_SAMPLES,
        SAMPLE_RATE
    );

    for (int i = 0; i < MFCC_COEFFS; i++) {
        mfcc_out[i] = output.values[i];
    }
}
