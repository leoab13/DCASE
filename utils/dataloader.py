import pandas as pd
import torch
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    def __init__(self, csv_file):
        """
        Args:
            csv_file (str): Ruta al archivo CSV con características y nombres de archivo.
        """
        # Leer el archivo CSV
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        """
        Returns:
            int: Número total de muestras en el dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Índice de la fila a recuperar.
        Returns:
            dict: Diccionario con características y etiqueta.
        """
        # Obtener la fila correspondiente
        row = self.data.iloc[idx]

        # Extraer las características (todas las columnas excepto 'filename')
        features = row.drop('filename').values.astype(float)
        features_tensor = torch.tensor(features, dtype=torch.float32)

        # Extraer la etiqueta desde el nombre del archivo
        filename = row['filename']
        label = self.extract_label_from_filename(filename)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return {'features': features_tensor, 'label': label_tensor}

    def extract_label_from_filename(self, filename):
        """
        Extrae la etiqueta desde el nombre del archivo.
        Args:
            filename (str): Nombre del archivo de audio.
        Returns:
            int: Etiqueta extraída.
        """
        # Dividir el nombre del archivo por guiones
        parts = filename.split('-')

        # Personaliza esta lógica según el patrón de las etiquetas
        if len(parts) > 1:
            # Por ejemplo, usar la segunda parte como etiqueta
            label = parts[1]
        else:
            raise ValueError(f"No se pudo extraer una etiqueta del archivo: {filename}")

        # Mapear etiquetas a números (puedes personalizar este mapeo)
        label_mapping = {
            "helsinki": 0,
            "barcelona": 1,
            "lisbon": 2,
            "london": 3,
            "milan": 4,
            "paris":5,
            "lyon":6
            # Agrega más etiquetas aquí si es necesario
        }
        return label_mapping.get(label, -1)  # Devuelve -1 si no se encuentra la etiqueta