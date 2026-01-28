import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from eml_net_model import EMLNet
import os

def test_model():
    # 1. Configuration
    model_path = "models/eml_net_hybrid.pth"
    image_path = "models/Datasets/Ueyes/images/003eb9.png"
    output_path = "test_saliency_map.png"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. Initialize Model
    print("Initializing model...")
    model = EMLNet().to(device)
    
    # 3. Load Weights
    print(f"Loading weights from {model_path}...")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle state dict structure
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
        
    # Handle DataParallel prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "")
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict)
    model.eval()
    print("Model loaded successfully.")
    
    # 4. Load & Preprocess Image
    print(f"Loading image from {image_path}...")
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return

    image = cv2.imread(image_path)
    if image is None:
        print("Error: Failed to read image")
        return
        
    h, w = image.shape[:2]
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((480, 640)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(img_rgb).unsqueeze(0).to(device)
    
    # 5. Inference
    print("Running inference...")
    with torch.no_grad():
        output = model(img_tensor)
        output = torch.sigmoid(output) # Sigmoid required as per training script output logic
        
    # 6. Post-process
    saliency_map = output.squeeze().cpu().numpy()
    
    # Resize back to original
    saliency_map = cv2.resize(saliency_map, (w, h), interpolation=cv2.INTER_CUBIC)
    
    # Normalize 0-1
    saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)
    
    # Save grayscale
    saliency_map_uint8 = (saliency_map * 255).astype(np.uint8)
    cv2.imwrite(output_path, saliency_map_uint8)
    print(f"Saved raw saliency map to {output_path}")
    
    # Generate Heatmap Overlay
    heatmap_colored = cv2.applyColorMap(saliency_map_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 0.6, heatmap_colored, 0.4, 0)
    cv2.imwrite("test_heatmap_overlay.jpg", overlay)
    print("Saved heatmap overlay to test_heatmap_overlay.jpg")
    
    print("\nTest Complete!")

if __name__ == "__main__":
    test_model()
