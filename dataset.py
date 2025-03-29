import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch

class AudioDataset(Dataset):
    def __init__(self, csv_file, processed_data_dir, transform=None):
        """
        Args:
            csv_file (str): Path to the CSV file with annotations.
            processed_data_dir (str): Directory with processed audio features.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.annotations = pd.read_csv(csv_file)
        self.processed_data_dir = processed_data_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get the filename and label from the CSV
        filename = self.annotations.iloc[idx, 0]
        label = self.annotations.iloc[idx, 1]  # Assuming the second column contains labels

        # Load the processed features
        file_path = os.path.join(self.processed_data_dir, filename)
        features = np.load(file_path)  # Assuming features are saved as .npy files

        sample = {'features': features, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

# Example usage
if __name__ == "__main__":
    # Paths to the CSV and processed data directory
    csv_path = "dataset/splits/train.csv"  # Update with the correct path
    processed_data_dir = "data/processed/"

    # Create the dataset and DataLoader
    dataset = AudioDataset(csv_file=csv_path, processed_data_dir=processed_data_dir)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Iterate through the DataLoader
    for batch in dataloader:
        print(batch['features'].shape, batch['label'])