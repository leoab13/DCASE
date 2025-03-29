from dataset import AudioDataset
from torch.utils.data import DataLoader

# Ruta al archivo CSV con las características y etiquetas
csv_path = "data/processed/audio_features1.csv"

# Crear el dataset
dataset = AudioDataset(csv_file=csv_path)

# Crear el DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Iterar sobre los lotes de datos
for batch_idx, batch in enumerate(dataloader):
    # Imprimir las dimensiones de las características y etiquetas
    print(f"Lote {batch_idx + 1}:")
    print(f"  Dimensiones de las características: {batch['features'].shape}")
    print(f"  Etiquetas: {batch['label']}")
    
    # Si necesitas detenerte después de algunos lotes para depuración
    if batch_idx == 2:  # Por ejemplo, detenerse después de 3 lotes
        break