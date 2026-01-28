import os
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

# Import the model architecture
try:
    from eml_net_model import EMLNet
except ImportError:
    # Handle case where file might not be found if running from different context
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from eml_net_model import EMLNet

class SaliencyEngine:
    def __init__(self, models_dir="models"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.models_dir = models_dir
        
        # Define paths
        self.encoder_path = os.path.join(models_dir, "resnet50_places365.pth.tar")
        self.decoder_path = os.path.join(models_dir, "eml_net_decoder.pth")
        
        self.load_model()
        
    def load_model(self):
        """
        Load EML-NET model weights from the trained hybrid model.
        """
        print(f"Initializing EML-NET Saliency Engine on {self.device}...")
        
        # Path to the trained hybrid model
        self.hybrid_model_path = os.path.join(self.models_dir, "eml_net_hybrid.pth")
        
        if not os.path.exists(self.hybrid_model_path):
             print(f"Warning: Trained model not found at '{self.hybrid_model_path}'.")
             print("Saliency prediction will fallback to Gaussian blobs.")
             self.model = None
             return
             
        try:
            # Initialize Model
            model = EMLNet()
            
            # Load the unified model weights
            print(f"Loading model from {self.hybrid_model_path}...")
            checkpoint = torch.load(self.hybrid_model_path, map_location=self.device)
            
            # Handle state dict structure
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Handle DataParallel prefix and load
            clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(clean_state_dict, strict=False)
            
            model.to(self.device)
            model.eval()
            self.model = model
            print("EML-NET model loaded successfully.")
            
        except Exception as e:
            print(f"Error loading EML-NET: {e}")
            print("Fallback to Gaussian blobs.")
            self.model = None

    def predict(self, image_numpy):
        """
        Predict saliency map using sliding window for tall images to preserve details.
        
        Args:
            image_numpy: Input image (BGR) from OpenCV.
            
        Returns:
            np.ndarray: Saliency map (0-255 uint8)
        """
        if self.model is None:
            return None
            
        try:
            h, w = image_numpy.shape[:2]
            
            # Model expects 4:3 input (640x480)
            model_w, model_h = 640, 480
            
            # Check if image is "tall" (e.g. height > 1.2 * width)
            # If so, use sliding window to avoid squashing
            if h > 1.2 * w:
                print(f"  Detected tall image ({w}x{h}). Using sliding window inference...")
                
                # Determine window size in the source image
                # We want a 4:3 window that fits the width
                window_w = w
                window_h = int(w * (model_h / model_w)) # Maintain aspect ratio
                
                stride = int(window_h * 0.5) # 50% overlap for smoother blending
                
                # Accumulators
                full_saliency = np.zeros((h, w), dtype=np.float32)
                counts = np.zeros((h, w), dtype=np.float32)
                
                # Sliding window loop
                for y in range(0, h, stride):
                    # Define crop
                    end_y = min(y + window_h, h)
                    start_y = max(0, end_y - window_h) # Ensure fixed height at bottom
                    
                    crop = image_numpy[start_y:end_y, 0:w]
                    
                    # Predict on crop
                    pred_map = self._predict_single(crop) # Returns 0-1 float
                    
                    # Resize back to crop size (should be minor)
                    pred_map_resized = cv2.resize(pred_map, (w, end_y - start_y))
                    
                    # Apply Gaussian window weighting to reduce edge artifacts
                    # Simple center weighting
                    # Y-weighting
                    weight_y = np.hanning(end_y - start_y)
                    # X-weighting
                    weight_x = np.hanning(w)
                    weight_map = np.outer(weight_y, weight_x)
                    
                    full_saliency[start_y:end_y, 0:w] += pred_map_resized * weight_map
                    counts[start_y:end_y, 0:w] += weight_map
                
                # Average
                final_map = np.divide(full_saliency, counts + 1e-8)
                
            else:
                # Standard inference for normal images
                final_map = self._predict_single(image_numpy)
                # Resize to original
                final_map = cv2.resize(final_map, (w, h), interpolation=cv2.INTER_CUBIC)

            # Normalize final output
            final_map = (final_map - final_map.min()) / (final_map.max() - final_map.min() + 1e-8)
            return (final_map * 255).astype(np.uint8)

        except Exception as e:
            print(f"Error during saliency prediction: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _predict_single(self, image_numpy):
        """Helper for single-pass prediction. Returns 0-1 float map."""
        img_rgb = cv2.cvtColor(image_numpy, cv2.COLOR_BGR2RGB)
        
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((480, 640)), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        img_tensor = transform(img_rgb).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(img_tensor)
            
        saliency_map = output.squeeze().cpu().numpy() # Raw logits or sigmoid
        
        # Normalize 0-1 locally
        saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)
        
        return saliency_map
