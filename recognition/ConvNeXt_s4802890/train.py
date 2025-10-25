import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from modules import ConvNeXt
from dataset import get_data_loaders

def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(loader, desc='Training'):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Validating'):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate_model(model, loader, device):
    """Evaluate model and return predictions and labels"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Evaluating'):
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return np.array(all_preds), np.array(all_labels)

def plot_training_history(train_losses, train_accs, val_losses, val_accs):
    """Plot and save training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss plot
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, train_accs, 'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-', label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Training curves saved to 'training_history.png'")

def plot_confusion_matrix(cm, class_names=['NC', 'AD']):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.title('Confusion Matrix - ADNI Test Set', fontsize=14)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Confusion matrix saved to 'confusion_matrix.png'")

def main():
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Hyperparameters
    batch_size = 32
    learning_rate = 5e-5
    num_epochs = 150
    
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
    class_weights = torch.tensor([1.0, 1.5]).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1, weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.05)

    print("\nTraining configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Optimizer: AdamW with weight decay 0.01")
    
    # Training loop
    print("\nStarting training...\n")
    best_acc = 0.0
    
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, test_loader, criterion, device)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'convnext_adni_best.pth')
            print(f"  Saved best model with accuracy: {best_acc:.4f}")
        
        print()
    
    print(f"Training complete! Best validation accuracy: {best_acc:.4f}")

    print("\nGenerating training plots...")
    plot_training_history(train_losses, train_accs, val_losses, val_accs)
    
    # Final evaluation on test set
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load('convnext_adni_best.pth'))
    predictions, labels = evaluate_model(model, test_loader, device)
    
    # Calculate metrics
    accuracy = (predictions == labels).mean()
    cm = confusion_matrix(labels, predictions)
    class_names = ['NC', 'AD']
    report = classification_report(labels, predictions, target_names=class_names, digits=4)
    
    # Print results
    print("\n" + "="*50)
    print("FINAL TEST RESULTS")
    print("="*50)
    print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Target: ≥0.8000 (80.00%)")
    if accuracy >= 0.8:
        print("✓ TARGET ACHIEVED!")
    else:
        print("✗ Target not reached")
    
    print("\nClassification Report:")
    print(report)
    
    print("\nConfusion Matrix:")
    print(cm)
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, class_names)
    
    # Save final model
    torch.save(model.state_dict(), 'convnext_adni_final.pth')
    print("\nFinal model saved to 'convnext_adni_final.pth'")

if __name__ == "__main__":
    main()
