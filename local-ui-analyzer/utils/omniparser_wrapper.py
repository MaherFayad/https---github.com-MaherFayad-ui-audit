"""
OmniParser V2 Wrapper for Local UI Analyzer.

Provides pixel-perfect bounding boxes for UI elements using Microsoft's
OmniParser V2 (YOLO icon detection + Florence captioning).

We bypass the heavy PaddleOCR dependency by patching the import before
loading OmniParser's utils module.
"""
import sys
import os
import types
import torch
import cv2
import base64
import numpy as np
from pathlib import Path
from PIL import Image

# ---- Patch: Create a fake paddleocr module so OmniParser's utils.py doesn't crash ----
# OmniParser eagerly imports PaddleOCR at module level, but we only use EasyOCR.
fake_paddleocr = types.ModuleType("paddleocr")
class _FakePaddleOCR:
    def __init__(self, *args, **kwargs):
        pass
    def ocr(self, *args, **kwargs):
        return [[]]
fake_paddleocr.PaddleOCR = _FakePaddleOCR
sys.modules["paddleocr"] = fake_paddleocr

# Add models/OmniParser to sys.path so we can import its modules
OMNIPARSER_DIR = Path(__file__).parent.parent / "models" / "OmniParser"
if str(OMNIPARSER_DIR) not in sys.path:
    sys.path.insert(0, str(OMNIPARSER_DIR))

# Now safe to import OmniParser
from util.utils import get_yolo_model, get_caption_model_processor, check_ocr_box, get_som_labeled_img


class LocalOmniParser:
    """Singleton wrapper for OmniParser V2 to ensure models are only loaded into VRAM once."""
    _instance = None
    
    def __init__(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        weights_dir = OMNIPARSER_DIR / 'weights'
        
        som_model_path = str(weights_dir / 'icon_detect' / 'model.pt')
        caption_model_path = str(weights_dir / 'icon_caption')
        
        print("Initializing Microsoft OmniParser V2 models...")
        self.som_model = get_yolo_model(model_path=som_model_path)
        self.caption_model_processor = get_caption_model_processor(
            model_name='florence2',
            model_name_or_path=caption_model_path,
            device=device
        )
        print("OmniParser V2 loaded successfully.")
        
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def predict(self, image_np: np.ndarray) -> list[dict]:
        """
        Parses a BGR numpy image and returns UI element bounding boxes.
        
        Returns:
            List of dicts: [{x, y, w, h, label, type}, ...]
        """
        height, width = image_np.shape[:2]
        
        # Convert BGR to RGB PIL Image for OmniParser
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Run OCR with EasyOCR (not PaddleOCR)
        print("  OmniParser: Running EasyOCR text detection...")
        (ocr_text, ocr_bbox), _ = check_ocr_box(
            pil_image, 
            display_img=False, 
            output_bb_format='xyxy', 
            easyocr_args={'text_threshold': 0.8}, 
            use_paddleocr=False
        )
        
        # Overlay ratio for text scaling
        box_overlay_ratio = max(width, height) / 3200
        draw_bbox_config = {
            'text_scale': 0.8 * box_overlay_ratio,
            'text_thickness': max(int(2 * box_overlay_ratio), 1),
            'text_padding': max(int(3 * box_overlay_ratio), 1),
            'thickness': max(int(3 * box_overlay_ratio), 1),
        }
        
        # Run YOLO + Florence captioning
        print("  OmniParser: Detecting UI elements with YOLO + Florence...")
        _, label_coordinates, parsed_content_list = get_som_labeled_img(
            pil_image, 
            self.som_model, 
            BOX_TRESHOLD=0.05,
            output_coord_in_ratio=True, 
            ocr_bbox=ocr_bbox,
            draw_bbox_config=draw_bbox_config, 
            caption_model_processor=self.caption_model_processor, 
            ocr_text=ocr_text,
            use_local_semantics=True, 
            iou_threshold=0.7, 
            scale_img=False, 
            batch_size=128
        )
        
        # Convert to our box format {x, y, w, h, label, type}
        boxes = []
        for elem in parsed_content_list:
            bbox = elem['bbox']  # [x1, y1, x2, y2] normalized 0-1
            
            x_min = int(bbox[0] * width)
            y_min = int(bbox[1] * height)
            x_max = int(bbox[2] * width)
            y_max = int(bbox[3] * height)
            
            box_dict = {
                'x': max(0, x_min),
                'y': max(0, y_min),
                'w': max(1, x_max - x_min),
                'h': max(1, y_max - y_min),
                'label': elem.get('content') or 'UI Element',
                'type': elem.get('type', 'icon'),
            }
            boxes.append(box_dict)
            
        print(f"  OmniParser: Found {len(boxes)} UI elements.")
        return boxes
