import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch

class AudioDataset(Dataset):
    def __init__(self, csv_file, processed_data_dir):
        """
        Args:
            csv_file (str): Path to the CSV file with annotations.
            processed_data_dir (str): Directory with processed audio features.
        """
        self.annotations = pd.read_csv(csv_file)
        self.processed_data_dir = os.path.normpath(processed_data_dir)  # Normalize the path

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get the filename and label from the CSV
        filename = self.annotations.iloc[idx]['filename'].strip()
        file_path = os.path.join(self.processed_data_dir, filename)        
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Processed file not found: {file_path}")
        print(f"Loading file: {file_path}")
        features = np.load(file_path, allow_pickle=True)

        # Convert label to tensor
        label = torch.tensor(label, dtype=torch.long)

        sample = {'features': torch.tensor(features, dtype=torch.float32), 'label': label}

        return sample