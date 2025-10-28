import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
import itertools
import numpy as np

from modules import ConvNeXt
from dataset import get_data_loaders

def train_and_evaluate(learning_rate, weight_decay, dropout, device, num_epochs=10):
    """Train model with given hyperparameters and return average validation accuracy"""
    
    train_loader, test_loader = get_data_loaders(batch_size=32, num_workers=4)
    
    model = ConvNeXt(in_channels=1, num_classes=2)
    model = model.to(device)
    
    class_weights = torch.tensor([1.0, 1.5]).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1, weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    val_accs = []
    
    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_acc = correct / total
        val_accs.append(val_acc)
    
    avg_acc = np.mean(val_accs[-3:])
    return avg_acc