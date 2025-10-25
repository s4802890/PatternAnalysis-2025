import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

"""
ADNI Dataset loader for Alzheimer's Disease classification
"""
class ADNIDataset(Dataset):
    """
    ADNI dataset for binary classification: AD vs NC
    Works with JPEG/PNG images
    """
    def __init__(self, data_dir='/home/groups/comp3710/ADNI/AD_NC', 
                 split='train', transform=None):
        
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        split_dir = os.path.join(data_dir, split)
        
        # AD class (label = 1)
        ad_dir = os.path.join(split_dir, 'AD')
        self.ad_files = self._get_image_files(ad_dir)
        
        # NC class (label = 0)
        nc_dir = os.path.join(split_dir, 'NC')
        self.nc_files = self._get_image_files(nc_dir)
        
        # Combine files and labels
        self.files = self.ad_files + self.nc_files
        self.labels = [1] * len(self.ad_files) + [0] * len(self.nc_files)
        
        print(f"{split.upper()} - AD: {len(self.ad_files)}, NC: {len(self.nc_files)}")
        
    def _get_image_files(self, folder):
        """Get all image files (jpg, jpeg, png) from folder"""
        if not os.path.exists(folder):
            return []
        
        files = []
        for f in sorted(os.listdir(folder)):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                files.append(os.path.join(folder, f))
        return files
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img = Image.open(self.files[idx])
        if img.mode != 'L':
            img = img.convert('L')
        
        # Convert to tensor
        to_tensor = transforms.ToTensor()
        image = to_tensor(img)  # Shape: (1, H, W) for grayscale
        
        # Resize to 224x224 for ConvNeXt
        resize = transforms.Resize((224, 224))
        image = resize(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, self.labels[idx]
    
def get_data_loaders(batch_size=16, num_workers=4):
    """Create train and test data loaders"""
    
    # Training transforms (with augmentation)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=25),
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.9, 1.1)),  
        transforms.ColorJitter(brightness=0.4, contrast=0.4), 
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Test transforms (no augmentation)
    test_transform = transforms.Compose([
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Create datasets
    train_dataset = ADNIDataset(split='train', transform=train_transform)
    test_dataset = ADNIDataset(split='test', transform=test_transform)
    
    # Create loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers, 
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers, 
        pin_memory=True
    )
    
    return train_loader, test_loader
