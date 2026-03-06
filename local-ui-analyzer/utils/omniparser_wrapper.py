import sys
import os
import torch
import cv2
import base64
from pathlib import Path

# Add models/OmniParser to sys.path so we can import its modules seamlessly
OMNIPARSER_DIR = Path(__file__).parent.parent / "models" / "OmniParser"
if str(OMNIPARSER_DIR) not in sys.path:
    sys.path.insert(0, str(OMNIPARSER_DIR))

# Import the native wrapper class from the Microsoft OmniParser repo
from util.omniparser import Omniparser

class LocalOmniParser:
    """Singleton wrapper for OmniParser V2 to ensure models are only loaded into VRAM once."""
    _instance = None
    
    def __init__(self):
        # Configuration mapping to the downloaded weights structure
        config = {
            'som_model_path': str(OMNIPARSER_DIR / 'weights' / 'icon_detect' / 'model.pt'),
            'caption_model_name': 'florence2',
            'caption_model_path': str(OMNIPARSER_DIR / 'weights' / 'icon_caption_florence'),
            'BOX_TRESHOLD': 0.05
        }
        print("Initializing Microsoft OmniParser V2 models...")
        self.parser = Omniparser(config)
        print("OmniParser loaded successfully.")
        
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def predict(self, image_np) -> list[dict]:
        """
        Parses an image and returns a list of bounding boxes for UI elements.
        
        Returns:
            List of dicts containing: {x, y, w, h, label, type}
        """
        height, width = image_np.shape[:2]
        
        # Convert BGR numpy array to base64 for OmniParser consumption
        _, buffer = cv2.imencode('.png', image_np)
        img_b64 = base64.b64encode(buffer).decode('utf-8')
        
        # Calling the native parse method 
        print("OmniParser analyzing UI semantics...")
        _, parsed_content_list = self.parser.parse(img_b64)
        
        boxes = []
        for elem in parsed_content_list:
            bbox = elem['bbox'] # [x1, y1, x2, y2] normalized between 0 and 1
            
            x_min = int(bbox[0] * width)
            y_min = int(bbox[1] * height)
            x_max = int(bbox[2] * width)
            y_max = int(bbox[3] * height)
            
            # Format to match the rest of the analyze.py pipeline
            box_dict = {
                'x': max(0, x_min),
                'y': max(0, y_min),
                'w': max(1, x_max - x_min),
                'h': max(1, y_max - y_min),
                'label': elem.get('content') or 'UI Element',
                'type': elem.get('type') or 'icon'
            }
            boxes.append(box_dict)
            
        print(f"OmniParser found {len(boxes)} elements.")
        return boxes
