import os
import librosa

class AudioPreprocessor:
    @staticmethod
    def convert_to_mono_and_resample(input_file, output_file, sr=16000, mono=True):
        y, sr = librosa.load(input_file, sr=sr, mono=mono)
        librosa.output.write_wav(output_file, y, sr)
        print(f"Audio convertido y guardado en: {output_file}")

def convert_all_audio(input_dir, output_dir, sr=16000, mono=True):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.endswith(".wav"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            AudioPreprocessor.convert_to_mono_and_resample(input_path, output_path, sr, mono)

if __name__ == "__main__":
    input_dir = "data/raw"
    output_dir = "data/processed" 
    convert_all_audio(input_dir, output_dir)
