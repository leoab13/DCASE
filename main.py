from dataset import AudioDataset
from torch.utils.data import DataLoader

# Paths to the CSV and processed data directory
csv_path = "data/processed/audio_features1.csv"
processed_data_dir = '/home/javier/audio'

# Create the dataset and DataLoader
dataset = AudioDataset(csv_file=csv_path, processed_data_dir=processed_data_dir)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Iterate through the DataLoader
for batch in dataloader:
    print(batch['features'].shape, batch['label'])