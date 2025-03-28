import os
import librosa
import soundfile as sf

class AudioPreprocessor:
    @staticmethod
    def convert_to_mono_and_resample(input_file, output_file, sr=16000, mono=True):
        print(f"Procesando archivo: {input_file}")
        y, sr = librosa.load(input_file, sr=sr, mono=mono)
        sf.write(output_file, y, sr, subtype='PCM_16')
        print(f"Audio convertido y guardado en: {output_file}")

def convert_all_audio(input_dir, output_dir, sr=16000, mono=True):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Procesando archivos en: {input_dir}")

    for folder_name, subfolders, filenames in os.walk(input_dir):
        if 'audio' in folder_name.lower():
            print(f"Procesando carpeta de audio: {folder_name}")

            for filename in filenames:
                if filename.endswith(".wav"):
                    input_path = os.path.join(folder_name, filename)
                    
                    relative_path = os.path.relpath(input_path, input_dir)
                    output_path = os.path.join(output_dir, relative_path)

                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    
                    AudioPreprocessor.convert_to_mono_and_resample(input_path, output_path, sr, mono)

if __name__ == "__main__":
    input_dir = "data/raw"  
    output_dir = "data/processed" 
    
    print("Inicio de la conversión de audios...")

    convert_all_audio(input_dir, output_dir)

    print("Conversión completada.")
