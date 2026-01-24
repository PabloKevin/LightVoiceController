#pragma once

#include <cmath>
#include <vector>
#include "audio_recording.h"
//#include "dsps_convolution.h"
#include "esp_dsp.h"
#include <algorithm>
#include <cstring>

// Parámetros de normalización (de tu entrenamiento)
#define MFCC_MEAN -47.3197508271f
#define MFCC_STD 174.8682650662f

extern int working_len;

class AudioProcessor {
public:
    AudioProcessor() {}
    void normalize_audio(float* data, int len);
    void remove_dc_offset(float* data, int len);
    void audio_blur(float* data, int len, int kernel_size);
    void gaussian_blur(float* data, int len);
    void median_filter(float* data, int len, int kernel_size);
    void process_complete_pipeline(float* audio, int len);
    void apply_bandpass(float* data, int len, float fs, float lf_cut, float hf_cut);
};

void convolve_1d_same(float* input, int input_len, const float* kernel, int kernel_len, float* output);


class HighPassFilter {
private:
    // Coeficientes constantes para evitar modificaciones accidentales
    // Usamos 'static const' para que no ocupen RAM por cada instancia de la clase
    static const float b[8];
    static const float a[8];
    
    float w[8] = {0, 0, 0, 0, 0, 0, 0, 0}; // Buffer de estados (delay line)

public:
    HighPassFilter();

    /**
     * Procesa una sola muestra usando Direct Form II
     */
    float process(float x);

    /**
     * Procesa un array completo de muestras
     */
    void processArray(float* data, int len);
};


class BandPassFilter {
private:
    // Coeficientes constantes para evitar modificaciones accidentales
    // Usamos 'static const' para que no ocupen RAM por cada instancia de la clase
    static const float sos[6][6];
    
    float w[6][2];// Buffer de estados (delay line)

public:
    BandPassFilter();

    /**
     * Procesa una sola muestra usando Direct Form II
     */
    float process(float x);

    /**
     * Procesa un array completo de muestras
     */
    void processArray(float* data, int len);
};