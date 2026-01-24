#ifndef MFCC_H
#define MFCC_H

#include <Arduino.h>
#include <vector>
#include "esp_dsp.h"
#include "mel_data.h" // Este es el archivo generado por el script de arriba
#include <stdint.h>

class MFCCExtractor {
private:
    // Configuración coincidente con entrenamiento
    static const int N_MFCC = 13;
    static const int N_FFT = 512;
    static const int HOP_LENGTH = 256;
    static const int TARGET_WINDOWS = 64;

    // Constantes de normalización (AJUSTAR CON TUS VALORES DE PYTHON)
    const float MFCC_MEAN = -15.0f; 
    const float MFCC_STD = 25.0f;

    float* window;
    float* fft_buffer;

public:
    MFCCExtractor();
    ~MFCCExtractor();
    
    /**
     * @brief Extrae MFCCs y los formatea para el modelo de ML
     * @param audio Buffer de audio (float)
     * @param audio_len Longitud del audio
     * @param mfcc_output Buffer de salida de tamaño 13*64 (832 floats)
     */
    int extract_mfcc(const float* audio, int audio_len, float* mfcc_output);
};

#endif