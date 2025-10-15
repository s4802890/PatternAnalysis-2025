import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from modules import ConvNeXt
from dataset import get_data_loaders

def main():
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Hyperparameters
    batch_size = 32
    learning_rate = 1e-4
    num_epochs = 50
    
    # Data loaders
    print("Loading data...")
    train_loader, test_loader = get_data_loaders(batch_size=batch_size, num_workers=4)
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    
    # Model
    print("Initializing model...")
    model = ConvNeXt(in_channels=1, num_classes=2)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    print("\nTraining configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Optimizer: AdamW with weight decay 0.01")
    
if __name__ == "__main__":
    main()