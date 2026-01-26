from AudioProcessing import process_audio_wav
    

if __name__=="__main__":
    wav_file = "/home/pablo_kevin/Projects/LightVoiceController/DSP/DataSets/RawAudio_pruebas/rawAudio_prenderLuz_01.wav"
    output_path = "/home/pablo_kevin/Projects/LightVoiceController/DSP/DataSets/ProcessedAudio/"
    process_audio_wav(wav_file, output_path, plot=True)