"""
Saliency Bridge v2 - Inference Engine for EML-NET

Improvements:
- 50% tile overlap for seamless blending
- Multi-scale fusion (1.0x + 0.5x) for capturing fine and coarse features
- F-Pattern Prior (top-left + navigation weighting) for UI-specific attention

Author: Refactored for UX-Heatmap v2
"""

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
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from eml_net_model import EMLNet


class SaliencyEngine:
    """
    High-fidelity saliency prediction engine.
    
    Features:
    - 50% tile overlap for artifact-free blending
    - Multi-scale fusion for both text and structural elements
    - F-Pattern prior for UI-specific attention modeling
    """
    
    def __init__(self, models_dir: str = "models"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.models_dir = models_dir
        self.hybrid_model_path = os.path.join(models_dir, "eml_net_hybrid.pth")
        
        # Standard transform for EML-NET input
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self.load_model()
        
    def load_model(self):
        """Load EML-NET model weights from the trained hybrid model."""
        print(f"Initializing EML-NET v2 Saliency Engine on {self.device}...")
        
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
            checkpoint = torch.load(self.hybrid_model_path, map_location=self.device, weights_only=False)
            
            # Handle state dict structure
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Handle DataParallel prefix
            clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(clean_state_dict, strict=False)
            
            model.to(self.device)
            model.eval()
            self.model = model
            print("EML-NET v2 model loaded successfully.")
            
        except Exception as e:
            print(f"Error loading EML-NET: {e}")
            print("Fallback to Gaussian blobs.")
            self.model = None

    def predict(self, image_numpy: np.ndarray) -> np.ndarray:
        """
        Predict saliency map using high-fidelity multi-scale inference.
        
        Features:
        - 50% tile overlap (reduced from 25%) for seamless blending
        - Multi-scale fusion: combines 1.0x and 0.5x scale predictions
        - F-Pattern prior: weights top-left and navigation areas
        
        Args:
            image_numpy: Input image (BGR) from OpenCV.
            
        Returns:
            np.ndarray: Saliency map (0-255 uint8)
        """
        if self.model is None:
            return None
            
        try:
            h, w = image_numpy.shape[:2]
            
            # Run multi-scale inference
            scale1_map = self._predict_at_scale(image_numpy, scale=1.0)
            scale2_map = self._predict_at_scale(image_numpy, scale=0.5)
            
            # Resize scale2 back to original resolution
            scale2_map = cv2.resize(scale2_map, (w, h), interpolation=cv2.INTER_LANCZOS4)
            
            # Multi-scale fusion: weighted average
            # 0.7 weight on full resolution (fine details like text)
            # 0.3 weight on half resolution (coarse structural elements)
            fused_map = 0.7 * scale1_map + 0.3 * scale2_map
            
            # Apply F-Pattern Prior
            fused_map = self._apply_f_pattern_prior(fused_map)
            
            # Normalize final output
            fused_map = (fused_map - fused_map.min()) / (fused_map.max() - fused_map.min() + 1e-8)
            return (fused_map * 255).astype(np.uint8)

        except Exception as e:
            print(f"Error during saliency prediction: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _predict_at_scale(self, image_numpy: np.ndarray, scale: float = 1.0) -> np.ndarray:
        """
        Run sliding window inference at a given scale.
        
        Args:
            image_numpy: Input BGR image
            scale: Scale factor (1.0 = original, 0.5 = half resolution)
            
        Returns:
            Saliency map as float32 array (0-1 range)
        """
        h, w = image_numpy.shape[:2]
        
        # Rescale image if needed
        if scale != 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            image_scaled = cv2.resize(image_numpy, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        else:
            image_scaled = image_numpy
            new_w, new_h = w, h
        
        # Model expects 4:3 input (640x480)
        model_w, model_h = 640, 480
        
        # Determine window size
        if new_w >= 640:
            window_w = 640
        else:
            window_w = new_w
        
        window_h = int(window_w * (3.0 / 4.0))
        
        # Check if sliding window is needed
        needs_sliding = new_h > window_h
        
        if needs_sliding:
            return self._sliding_window_inference(image_scaled, window_w, window_h)
        else:
            return self._predict_single(image_scaled)

    def _sliding_window_inference(self, image_numpy: np.ndarray, 
                                   window_w: int, window_h: int) -> np.ndarray:
        """
        Sliding window inference with 50% overlap and Hanning blending.
        
        Args:
            image_numpy: Input BGR image
            window_w, window_h: Tile dimensions
            
        Returns:
            Fused saliency map (float32, 0-1 range)
        """
        h, w = image_numpy.shape[:2]
        
        # 50% Overlap Stride (increased from 25%)
        stride_x = int(window_w * 0.50)
        stride_y = int(window_h * 0.50)
        
        # Accumulators
        full_saliency = np.zeros((h, w), dtype=np.float32)
        weight_accumulator = np.zeros((h, w), dtype=np.float32)
        
        # Pre-compute 2D Hanning window
        hanning_y = np.hanning(window_h).astype(np.float32)
        hanning_x = np.hanning(window_w).astype(np.float32)
        hanning_2d = np.outer(hanning_y, hanning_x)
        
        # Sliding window loop
        y = 0
        while y < h:
            end_y = min(y + window_h, h)
            start_y = max(0, end_y - window_h)
            actual_h = end_y - start_y
            
            x = 0
            while x < w:
                end_x = min(x + window_w, w)
                start_x = max(0, end_x - window_w)
                actual_w = end_x - start_x
                
                # Extract tile
                tile = image_numpy[start_y:end_y, start_x:end_x]
                
                # Predict on tile
                pred_map = self._predict_single(tile)
                
                # Resize prediction back to tile size
                pred_map_resized = cv2.resize(
                    pred_map, 
                    (actual_w, actual_h), 
                    interpolation=cv2.INTER_LANCZOS4
                )
                
                # Get appropriate Hanning window
                if actual_h == window_h and actual_w == window_w:
                    weight_kernel = hanning_2d
                else:
                    edge_hanning_y = np.hanning(actual_h).astype(np.float32)
                    edge_hanning_x = np.hanning(actual_w).astype(np.float32)
                    weight_kernel = np.outer(edge_hanning_y, edge_hanning_x)
                
                # Accumulate
                full_saliency[start_y:end_y, start_x:end_x] += pred_map_resized * weight_kernel
                weight_accumulator[start_y:end_y, start_x:end_x] += weight_kernel
                
                if end_x >= w:
                    break
                x += stride_x
            
            if end_y >= h:
                break
            y += stride_y
        
        # Normalize
        return np.divide(full_saliency, weight_accumulator + 1e-8, dtype=np.float32)

    def _predict_single(self, image_numpy: np.ndarray) -> np.ndarray:
        """
        Single-pass prediction with Lanczos resampling.
        
        Args:
            image_numpy: Input BGR image
            
        Returns:
            Saliency map (float32, 0-1 range)
        """
        h, w = image_numpy.shape[:2]
        img_rgb = cv2.cvtColor(image_numpy, cv2.COLOR_BGR2RGB)
        
        # Resize to model input (640x480)
        img_resized = cv2.resize(img_rgb, (640, 480), interpolation=cv2.INTER_LANCZOS4)
        
        # Convert to tensor
        img_tensor = self.transform(img_resized).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(img_tensor)
            output = torch.sigmoid(output)  # Apply sigmoid for [0,1] range
            
        saliency_map = output.squeeze().cpu().numpy()
        
        # Resize back to original tile size
        saliency_map = cv2.resize(saliency_map, (w, h), interpolation=cv2.INTER_LANCZOS4)
        
        # Normalize locally
        saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)
        
        return saliency_map.astype(np.float32)

    def _apply_f_pattern_prior(self, saliency_map: np.ndarray) -> np.ndarray:
        """
        Apply F-Pattern prior for UI-specific attention modeling.
        
        Based on Nielsen Norman Group eye-tracking research:
        - Users scan horizontally across the top (navigation bar)
        - Then scan down the left side (sidebar / content start)
        - Creates an "F" shaped attention pattern
        
        Weights:
        - Top 15% (nav bar): 1.1x
        - Left 20%: 1.05x  
        - Top-left quadrant: additional 1.1x boost (combines with above)
        
        Args:
            saliency_map: Input saliency (float32)
            
        Returns:
            F-pattern weighted saliency map
        """
        h, w = saliency_map.shape
        
        # Create weight mask
        weight_mask = np.ones((h, w), dtype=np.float32)
        
        # Top 15% (navigation bar region)
        nav_height = int(h * 0.15)
        weight_mask[:nav_height, :] *= 1.1
        
        # Left 20% (sidebar / content start)
        sidebar_width = int(w * 0.20)
        weight_mask[:, :sidebar_width] *= 1.05
        
        # Top-left quadrant (additional boost for logo/nav intersection)
        top_half = h // 2
        left_half = w // 2
        weight_mask[:top_half, :left_half] *= 1.1
        
        # Apply weights
        weighted_map = saliency_map * weight_mask
        
        return weighted_map


if __name__ == "__main__":
    # Quick test
    engine = SaliencyEngine()
    
    # Create a dummy test image
    test_image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    
    result = engine.predict(test_image)
    if result is not None:
        print(f"Output shape: {result.shape}")
        print(f"Output range: [{result.min()}, {result.max()}]")
    else:
        print("Model not loaded, prediction returned None")
