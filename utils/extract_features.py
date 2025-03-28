import os
import librosa
import numpy as np
import pandas as pd
from scipy.stats import entropy
from librosa.feature import mfcc, spectral_contrast, spectral_bandwidth

class FeatureExtractor:
    @staticmethod
    def extract_features(y, sr):
        features = {}

        # MFCC (13 coeficientes)
        mfcc_features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc_features, axis=1)
        for i in range(13):
            features[f'mfcc_{i+1}'] = mfcc_mean[i]

        # Spectral Contrast
        spec_contrast = spectral_contrast(y=y, sr=sr)
        spec_contrast_mean = np.mean(spec_contrast, axis=1)
        for i in range(6):
            features[f'spec_contrast_{i+1}'] = spec_contrast_mean[i]

        # Spectral Bandwidth
        spec_bandwidth = spectral_bandwidth(y=y, sr=sr)
        features['spec_bandwidth'] = np.mean(spec_bandwidth)

        # Entropía
        features['entropy'] = entropy(np.histogram(y, bins=256)[0])

        # RMS (Root Mean Square)
        rms = librosa.feature.rms(y=y)
        features['rms'] = np.mean(rms)

        return features


def extract_features_from_folder(folder_path, output_dir, sr=16000):
    processed_data = []
    print(f"Procesando carpeta: {folder_path}")

    if not os.path.exists(folder_path):
        print(f"Error: El directorio {folder_path} no existe.")
        return
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".wav"):
            file_path = os.path.join(folder_path, filename)
            print(f"Procesando archivo: {filename}")

            try:
                y, sr = librosa.load(file_path, sr=sr, mono=True)
                features = FeatureExtractor.extract_features(y, sr)
                features['filename'] = filename
                processed_data.append(features)
            except Exception as e:
                print(f"Error al procesar el archivo {filename}: {e}")

    if processed_data:
        df = pd.DataFrame(processed_data)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{os.path.basename(folder_path)}_features.csv")
        df.to_csv(output_file, index=False)

        print(f"Datos procesados y guardados en {output_file}")
        return df
    else:
        print(f"No se procesaron archivos en {folder_path}.")
        return None


if __name__ == "__main__":
    folder_to_process = "C:/Users/quint/OneDrive/Escritorio/AUDIO DEL PELON/DCASE/data/processed/TAU-urban-acoustic-scenes-2022-mobile-development.audio.1/TAU-urban-acoustic-scenes-2022-mobile-development/audio"
    output_dir = "C:/Users/quint/OneDrive/Escritorio/AUDIO DEL PELON/DCASE/data/processed"  # Directorio donde se guardarán los CSVs

    print("Inicio de la conversión de audios...")
    
    extract_features_from_folder(folder_to_process, output_dir)

    print("Conversión completada.")
