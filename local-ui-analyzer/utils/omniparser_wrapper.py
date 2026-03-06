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
        Parses a BGR numpy image. Automatically chunks tall images.
        """
        if image_np is None:
            print("  OmniParser Warning: Received NoneType instead of image numpy array.")
            return []
            
        height, width = image_np.shape[:2]
        
        # If the image is already small enough, process it whole
        if width <= 640 and height <= 480:
            return self._predict_single(image_np, x_offset=0, y_offset=0)
            
        # Chunking logic for large or tall screens
        print(f"  OmniParser: Image is large ({width}x{height}px). Chunking to preserve detection accuracy...")
        chunk_w = 640
        chunk_h = 480
        
        overlap_w = int(chunk_w * 0.1) # 10% overlap
        overlap_h = int(chunk_h * 0.1)
        
        step_x = chunk_w - overlap_w
        step_y = chunk_h - overlap_h
        
        all_boxes = []
        y = 0
        chunk_idx = 1
        
        while y < height:
            end_y = min(y + chunk_h, height)
            
            # Skip very small final vertical chunks if they are just slivers
            if (end_y - y) < 100 and y > 0:
                break
                
            x = 0
            while x < width:
                end_x = min(x + chunk_w, width)
                
                # Skip very small final horizontal chunks
                if (end_x - x) < 100 and x > 0:
                    break
                    
                chunk = image_np[y:end_y, x:end_x]
                print(f"  OmniParser: Processing chunk {chunk_idx} (x={x} to {end_x}, y={y} to {end_y})...")
                
                # Get boxes for this chunk
                chunk_boxes = self._predict_single(chunk, x_offset=x, y_offset=y)
                all_boxes.extend(chunk_boxes)
                
                x += step_x
                chunk_idx += 1
                
            y += step_y
            
        # Simple NMS to remove overlapping boxes at chunk seams
        merged_boxes = self._merge_chunk_seam_boxes(all_boxes)
        print(f"  OmniParser: Found {len(merged_boxes)} UI elements after merging.")
        return merged_boxes
        
    def _merge_chunk_seam_boxes(self, boxes: list[dict], iou_threshold: float = 0.5) -> list[dict]:
        """Remove duplicates from chunk overlaps using simple IoU."""
        if not boxes:
            return []
            
        def calc_iou(b1, b2):
            x1 = max(b1['x'], b2['x'])
            y1 = max(b1['y'], b2['y'])
            x2 = min(b1['x'] + b1['w'], b2['x'] + b2['w'])
            y2 = min(b1['y'] + b1['h'], b2['y'] + b2['h'])
            
            if x2 <= x1 or y2 <= y1:
                return 0.0
                
            intersection = (x2 - x1) * (y2 - y1)
            area1 = b1['w'] * b1['h']
            area2 = b2['w'] * b2['h']
            union = area1 + area2 - intersection
            return intersection / union if union > 0 else 0.0

        # Sort by area (larger first)
        sorted_boxes = sorted(boxes, key=lambda b: b['w'] * b['h'], reverse=True)
        merged = []
        
        for box in sorted_boxes:
            is_dup = False
            for existing in merged:
                if calc_iou(box, existing) > iou_threshold:
                    is_dup = True
                    break
            if not is_dup:
                merged.append(box)
                
        return merged

    def _predict_single(self, image_np: np.ndarray, x_offset: int = 0, y_offset: int = 0) -> list[dict]:
        """
        Parses a single BGR numpy chunk.
        """
        if image_np is None:
            print("    OmniParser Warning: Received NoneType instead of image numpy array in _predict_single.")
            return []

        height, width = image_np.shape[:2]
        
        # Convert BGR to RGB PIL Image for OmniParser
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Run OCR with EasyOCR (not PaddleOCR)
        print(f"    OmniParser: Running EasyOCR text detection...")
        (ocr_text, ocr_bbox), _ = check_ocr_box(
            pil_image, 
            display_img=False, 
            output_bb_format='xyxy', 
            easyocr_args={'text_threshold': 0.8}, 
            use_paddleocr=False
        )
        
        # Prevent TypeError if OCR finds zero text
        if ocr_text is None:
            ocr_text = []
        if ocr_bbox is None:
            ocr_bbox = []
        
        # Overlay ratio for text scaling
        box_overlay_ratio = max(width, height) / 3200
        draw_bbox_config = {
            'text_scale': 0.8 * box_overlay_ratio,
            'text_thickness': max(int(2 * box_overlay_ratio), 1),
            'text_padding': max(int(3 * box_overlay_ratio), 1),
            'thickness': max(int(3 * box_overlay_ratio), 1),
        }
        
        # Run YOLO + Florence captioning
        print(f"    OmniParser: Detecting UI elements with YOLO + Florence...")
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
                'x': max(0, x_min) + x_offset,
                'y': max(0, y_min) + y_offset,
                'w': max(1, x_max - x_min),
                'h': max(1, y_max - y_min),
                'label': elem.get('content') or 'UI Element',
                'type': elem.get('type', 'icon'),
            }
            boxes.append(box_dict)
            
        print(f"    OmniParser: Found {len(boxes)} UI elements in chunk.")
        return boxes
