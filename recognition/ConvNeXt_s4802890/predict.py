import torch
from PIL import Image
from torchvision import transforms
import os

from modules import ConvNeXt

def predict_single_image(model, image_path, device):
    """
    Predict a single brain MRI image
    
    Args:
        model: Trained ConvNeXt model
        image_path: Path to image file
        device: torch device
    
    Returns:
        predicted_class: 'NC' or 'AD'
        confidence: Prediction confidence (0-1)
    """
    # Load and preprocess
    image = Image.open(image_path).convert('L')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
        predicted = output.argmax(1).item()
        confidence = probs[0][predicted].item()
    
    class_names = ['NC', 'AD']
    return class_names[predicted], confidence

def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}\n')
    
    # Load trained model
    print('Loading model...')
    model = ConvNeXt(in_channels=1, num_classes=2)
    model.load_state_dict(torch.load('convnext_adni_best.pth', map_location=device))
    model = model.to(device)
    print('Model loaded successfully!\n')
    
    # Example predictions
    data_dir = '/home/groups/comp3710/ADNI/AD_NC/test'
    
    # Get a few example images from each class
    ad_dir = os.path.join(data_dir, 'AD')
    nc_dir = os.path.join(data_dir, 'NC')
    
    ad_images = [os.path.join(ad_dir, f) for f in os.listdir(ad_dir) if f.endswith(('.jpg', '.jpeg', '.png'))][:3]
    nc_images = [os.path.join(nc_dir, f) for f in os.listdir(nc_dir) if f.endswith(('.jpg', '.jpeg', '.png'))][:3]
    
    print('Example Predictions:\n')
    print('='*60)
    
    for img_path in ad_images + nc_images:
        true_label = 'AD' if '/AD/' in img_path else 'NC'
        pred_label, confidence = predict_single_image(model, img_path, device)
        
        # Check if prediction is correct
        correct = '✓' if pred_label == true_label else '✗'
        
        print(f'Image: {os.path.basename(img_path)}')
        print(f'  True Label:      {true_label}')
        print(f'  Predicted:       {pred_label} ({confidence:.2%} confidence) {correct}')
        print()

if __name__ == '__main__':
    main()