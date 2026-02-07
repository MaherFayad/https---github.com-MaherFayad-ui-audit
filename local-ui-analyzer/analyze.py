"""
Local UI Analyzer - Analysis Engine
Generates attention heatmaps, contrast maps, focus maps, and accessibility reports
using Gemini's vision model for UI/UX analysis.
"""

import os
import re
import json
import base64
import time
from io import BytesIO
from urllib.parse import urlparse

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import requests
from google import genai
from PIL import Image

from dotenv import load_dotenv
from saliency_bridge import SaliencyEngine

# Load environment variables
load_dotenv()

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")
GEMINI_MODEL = "gemini-2.5-pro"

# Initialize Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)

# Initialize Saliency Engine
saliency_engine = SaliencyEngine()


def load_image(path_or_url: str) -> np.ndarray:
    """Load an image from a local path or URL."""
    parsed = urlparse(path_or_url)
    
    if parsed.scheme in ('http', 'https'):
        # Load from URL
        response = requests.get(path_or_url, timeout=30)
        response.raise_for_status()
        img_array = np.frombuffer(response.content, np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    else:
        # Load from local path
        if not os.path.exists(path_or_url):
            raise FileNotFoundError(f"Image not found: {path_or_url}")
        image = cv2.imread(path_or_url)
    
    if image is None:
        raise ValueError(f"Failed to load image: {path_or_url}")
    
    return image


def image_to_base64(image: np.ndarray) -> str:
    """Convert a numpy image to base64 string for Ollama."""
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')


def chunk_image(image: np.ndarray, saliency_map: np.ndarray = None, 
                chunk_size: tuple = (800, 600), 
                overlap: float = 0.2) -> list[dict]:
    """
    Split image into overlapping chunks for better VLM analysis.
    
    Args:
        image: Input image
        chunk_size: (width, height) of each chunk
        overlap: Percentage overlap between chunks (0.0 to 0.5)
        
    Returns:
        List of dicts with 'image', 'x_offset', 'y_offset', 'row', 'col'
    """
    height, width = image.shape[:2]
    chunk_w, chunk_h = chunk_size
    
    # Calculate step size (considering overlap)
    step_x = int(chunk_w * (1 - overlap))
    step_y = int(chunk_h * (1 - overlap))
    
    chunks = []
    
    # Define ranges - valid single chunk if dimension fits
    x_range = range(0, width, step_x) if width > chunk_w else [0]
    y_range = range(0, height, step_y) if height > chunk_h else [0]
    
    row = 0
    for y in y_range:
        col = 0
        for x in x_range:
            # Extract chunk (handle edge cases)
            end_x = min(x + chunk_w, width)
            end_y = min(y + chunk_h, height)
            chunk = image[y:end_y, x:end_x]
            
            # Skip very small chunks
            if chunk.shape[0] < 100 or chunk.shape[1] < 100:
                continue
                
            chunks.append({
                'image': chunk,
                'x_offset': x,
                'y_offset': y,
                'row': row,
                'col': col,
                'height': end_y - y,
                'saliency_map': saliency_map[y:end_y, x:end_x] if saliency_map is not None else None
            })
            col += 1
        row += 1
    
    return chunks


def analyze_single_chunk(chunk_data: dict) -> list[dict]:
    """
    Analyze a single image chunk using Gemini Vision.
    Returns boxes with coordinates adjusted to full image.
    """
    chunk = chunk_data['image']
    x_offset = chunk_data['x_offset']
    y_offset = chunk_data['y_offset']
    height, width = chunk.shape[:2]
    
    # Prepare inputs for Gemini
    contents = []
    
    # 1. UI Screenshot
    chunk_rgb = cv2.cvtColor(chunk, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(chunk_rgb)
    
    # 2. Saliency Heatmap (if available) - This is the key change!
    heatmap_context = ""
    if chunk_data.get('saliency_map') is not None:
        # Apply colormap to make it easier for Gemini to see "hot" areas
        saliency_chunk = chunk_data['saliency_map']
        heatmap_colored = cv2.applyColorMap(saliency_chunk, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_RGB2BGR)
        pil_heatmap = Image.fromarray(heatmap_rgb)
        
        contents = [pil_image, pil_heatmap]
        heatmap_context = """
**CRITICAL INSTRUCTION**: You are provided with TWO images:
1. The UI Screenshot (Visual).
2. The Saliency Heatmap (Ground Truth). [Red/Yellow = High Attention, Blue = Low].

**YOUR TASK**:
- Use the Saliency Heatmap to validte your predictions.
- IF an area is RED/YELLOW in the heatmap, you MUST identify the UI element at that location.
- IF an area is BLUE, do not assign a "focal" type unless it is a critical navigation element.
- Identify the UI elements that correspond to the high-attention hotspots.
"""
    else:
        contents = [pil_image]

    prompt = f"""Act as a predictive eye-tracking model.
Analyze this UI section ({width}x{height} pixels) to predict user attention fixations.
{heatmap_context}

Principles of Visual Focus:
1. **High Contrast**: Bright/dark elements stand out (CTAs, buttons).
2. **Face Bias**: Human faces immediately capture gaze.
3. **Typography**: Large, bold headings attract eyes.
4. **Imagery**: meaningful photos or illustrations.
5. **Form**: Inputs and active elements.
6. **Context Bias**: Adapt to page type:
   - **E-commerce**: Product Interest > Navigation (Product image is king).
   - **Blog/Article**: Headline > Author face > Body text.
   - **Landing Page**: Value Prop > Primary CTA > Hero Image.

Task:
- Identify TWO types of attention elements (Hybrid Saliency):
  1. "focal": Distinct objects (Buttons, Faces, Products, Headings). High attention.
  2. "structural": Large context areas (Hero sections, Containers, Navigation bars). Lower attention but guides flow.
- Be granular. Separate the "Hero Title" (focal) from the "Hero Container" (structural).

Return ONLY a JSON array:
[
    {{"x": 10, "y": 20, "w": 100, "h": 50, "label": "Primary CTA", "type": "focal"}},
    {{"x": 0, "y": 0, "w": 300, "h": 100, "label": "Hero Section", "type": "structural"}}
]

Where x,y are relative to the top-left of this image chunk.
Return ONLY raw JSON. No markdown formatting.
"""

    max_retries = 3
    base_delay = 2
    
    for attempt in range(max_retries):
        try:
            full_contents = [prompt] + contents
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=full_contents
            )
            response_text = response.text.strip()
            
            # Extract JSON from response
            json_match = re.search(r'\[[\s\S]*?\]', response_text)
            if json_match:
                boxes = json.loads(json_match.group())
                adjusted_boxes = []
                for box in boxes:
                    if all(k in box for k in ['x', 'y', 'w', 'h']):
                        adjusted_boxes.append({
                            'x': max(0, int(box['x']) + x_offset),
                            'y': max(0, int(box['y']) + y_offset),
                            'w': max(1, int(box['w'])),
                            'h': max(1, int(box['h'])),
                            'label': box.get('label', 'UI Element'),
                            'chunk': f"r{chunk_data['row']}c{chunk_data['col']}"
                        })
                return adjusted_boxes
            else:
                print(f"  Warning: No valid JSON found in chunk response (Attempt {attempt+1}/{max_retries})")
                
        except Exception as e:
            print(f"  Warning: Chunk analysis failed (Attempt {attempt+1}/{max_retries}): {e}")
            if "429" in str(e) or "503" in str(e) or "500" in str(e) or "overloaded" in str(e).lower():
                time.sleep(base_delay * (2 ** attempt))  # Exponential backoff
                continue
            else:
                break # Break on non-transient errors
        
        # If we got here but failed to parse/extract, maybe retry? 
        # Usually parsing error isn't transient unless model output was cut off.
        # Let's retry on parsing failure too if we have attempts left.
        # But for now, we only retry on Exception or explicitly coded conditions.
    
    print(f"  Error: Failed to analyze chunk after {max_retries} attempts.")
    return []


def merge_chunk_boxes(all_boxes: list[dict], iou_threshold: float = 0.5) -> list[dict]:
    """
    Hierarchy-Aware NMS: Merge overlapping boxes while preserving focal elements inside structural containers.
    
    Key improvement: If a smaller "focal" box is contained within a larger "structural" box,
    we KEEP both boxes (e.g., a CTA button inside a hero section).
    
    Args:
        all_boxes: List of boxes with 'x', 'y', 'w', 'h', 'type' (focal/structural)
        iou_threshold: IoU threshold for standard deduplication
        
    Returns:
        Deduplicated list of boxes preserving hierarchy
    """
    if not all_boxes:
        return []
    
    def calc_iou(box1, box2):
        """Calculate Intersection over Union."""
        x1 = max(box1['x'], box2['x'])
        y1 = max(box1['y'], box2['y'])
        x2 = min(box1['x'] + box1['w'], box2['x'] + box2['w'])
        y2 = min(box1['y'] + box1['h'], box2['y'] + box2['h'])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = box1['w'] * box1['h']
        area2 = box2['w'] * box2['h']
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def calc_containment(smaller, larger):
        """Calculate how much of smaller box is contained in larger box."""
        x1 = max(smaller['x'], larger['x'])
        y1 = max(smaller['y'], larger['y'])
        x2 = min(smaller['x'] + smaller['w'], larger['x'] + larger['w'])
        y2 = min(smaller['y'] + smaller['h'], larger['y'] + larger['h'])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        smaller_area = smaller['w'] * smaller['h']
        
        return intersection / smaller_area if smaller_area > 0 else 0.0
    
    def is_focal_in_structural(box_a, box_b):
        """
        Check if box_a is a focal element properly contained in structural box_b.
        Returns True if we should KEEP box_a despite high IoU.
        """
        type_a = box_a.get('type', 'focal')
        type_b = box_b.get('type', 'focal')
        area_a = box_a['w'] * box_a['h']
        area_b = box_b['w'] * box_b['h']
        
        # box_a must be smaller (the "inner" element)
        if area_a >= area_b:
            return False
            
        # box_a is focal, box_b is structural
        if type_a == 'focal' and type_b == 'structural':
            # Check if box_a is 80%+ contained within box_b
            containment = calc_containment(box_a, box_b)
            if containment >= 0.8:
                return True
        
        return False
    
    # Sort by area (larger first)
    sorted_boxes = sorted(all_boxes, key=lambda b: b['w'] * b['h'], reverse=True)
    merged = []
    
    for box in sorted_boxes:
        is_duplicate = False
        
        for existing in merged:
            iou = calc_iou(box, existing)
            
            if iou > iou_threshold:
                # Check hierarchy: should we preserve focal inside structural?
                if is_focal_in_structural(box, existing):
                    # Keep the focal box even though it overlaps with structural
                    is_duplicate = False
                    break
                elif is_focal_in_structural(existing, box):
                    # Existing is focal inside this structural - don't mark as duplicate
                    is_duplicate = False
                    continue
                else:
                    # Standard NMS: mark as duplicate
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            merged.append(box)
    
    return merged


def get_attention_boxes(image: np.ndarray, saliency_map: np.ndarray = None, use_chunking: bool = True) -> list[dict]:
    """
    Query Ollama to identify eye-catching UI elements.
    Uses chunking strategy for better results with small VLMs.
    
    Args:
        image: Input image
        saliency_map: Optional EML-NET heatmap to guide detection
        use_chunking: If True, split image into chunks first
        
    Returns:
        List of bounding boxes [x, y, w, h, label]
    """
    height, width = image.shape[:2]
    
    # For small images, don't chunk
    if not use_chunking or (width <= 1920 and height <= 4000):
        return _get_attention_boxes_single(image)
    
    # Chunk the image for better accuracy
    print(f"  Chunking image ({width}x{height})...")
    chunks = chunk_image(image, saliency_map=saliency_map, chunk_size=(1921, 4000), overlap=0.1)
    print(f"  Created {len(chunks)} chunks")
    
    # Analyze each chunk with rate limiting
    all_boxes = []
    for i, chunk_data in enumerate(chunks):
        print(f"  Analyzing chunk {i+1}/{len(chunks)} (row {chunk_data['row']}, col {chunk_data['col']})...")
        boxes = analyze_single_chunk(chunk_data)
        all_boxes.extend(boxes)
        print(f"    Found {len(boxes)} elements")
        # Rate limit: wait between API calls to avoid 429
        if i < len(chunks) - 1:
            time.sleep(10)
    
    # Merge overlapping boxes
    print(f"  Merging {len(all_boxes)} boxes...")
    merged_boxes = merge_chunk_boxes(all_boxes, iou_threshold=0.3)
    print(f"  Final: {len(merged_boxes)} unique attention areas")
    
    # Clamp to image bounds
    validated = []
    for box in merged_boxes:
        validated.append({
            'x': max(0, min(box['x'], width - 1)),
            'y': max(0, min(box['y'], height - 1)),
            'w': max(1, min(box['w'], width - box['x'])),
            'h': max(1, min(box['h'], height - box['y'])),
            'label': box.get('label', 'UI Element')
        })
    
    if not validated:
        # Fallback if nothing found
        return [{
            'x': width // 4, 'y': height // 4,
            'w': width // 2, 'h': height // 2,
            'label': 'Center region (fallback)'
        }]
    
    return validated


def _get_attention_boxes_single(image: np.ndarray) -> list[dict]:
    """Single-image analysis using Gemini Vision (for small images)."""
    height, width = image.shape[:2]
    
    # Convert OpenCV to PIL
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    
    prompt = f"""Act as a predictive eye-tracking model.
Analyze this UI screenshot ({width}x{height} pixels) to predict user attention fixations.

Principles of Visual Focus:
1. **High Contrast**: Bright/dark elements stand out (CTAs, buttons).
2. **Face Bias**: Human faces immediately capture gaze.
3. **Typography**: Large, bold headings attract eyes.
4. **Imagery**: meaningful photos or illustrations.
5. **Form**: Inputs and active elements.
6. **Context Bias**: Adapt to page type:
   - **E-commerce**: Product Interest > Navigation (Product image is king).
   - **Blog/Article**: Headline > Author face > Body text.
   - **Landing Page**: Value Prop > Primary CTA > Hero Image.

Task:
- Identify TWO types of attention elements (Hybrid Saliency):
  1. "focal": Distinct objects (Buttons, Faces, Products, Headings).
  2. "structural": Layout blocks (Containers, Regions).
  
Return ONLY a JSON array:
[
    {{"x": 10, "y": 20, "w": 100, "h": 50, "label": "CTA", "type": "focal"}},
    {{"x": 0, "y": 0, "w": 1920, "h": 500, "label": "Hero", "type": "structural"}}
]

Where x,y are from the top-left corner.
Return ONLY raw JSON. No markdown formatting."""



    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[prompt, pil_image]
        )
        response_text = response.text.strip()
        
        json_match = re.search(r'\[[\s\S]*?\]', response_text)
        if json_match:
            boxes = json.loads(json_match.group())
            validated = []
            for box in boxes:
                if all(k in box for k in ['x', 'y', 'w', 'h']):
                    validated.append({
                        'x': max(0, min(int(box['x']), width - 1)),
                        'y': max(0, min(int(box['y']), height - 1)),
                        'w': max(1, min(int(box['w']), width - int(box.get('x', 0)))),
                        'h': max(1, min(int(box['h']), height - int(box.get('y', 0)))),
                        'label': box.get('label', 'UI Element')
                    })
            if validated:
                return validated
    except Exception as e:
        print(f"Warning: Gemini attention detection failed: {e}")
    
    return [{
        'x': width // 4, 'y': height // 4,
        'w': width // 2, 'h': height // 2,
        'label': 'Center region (fallback)'
    }]


def get_saliency_map(image: np.ndarray, boxes: list[dict]) -> np.ndarray:
    """
    Get saliency map from EML-NET or fallback to Gaussian blobs.
    Returns 0-1 float32 map.
    """
    # Try EML-NET
    real_map = saliency_engine.predict(image)
    if real_map is not None:
        return real_map.astype(np.float32) / 255.0
    
    # Fallback
    return generate_eml_heatmap_mask(image.shape[:2], boxes)


def generate_attention_heatmap(image: np.ndarray, boxes: list[dict], 
                              saliency_map: np.ndarray = None) -> np.ndarray:
    """
    Generate a Gaussian-blurred heatmap overlay based on attention boxes.
    Returns the image with heatmap overlay.
    
    Uses heavy blur to create soft, blobby attention pools similar to 
    Attention Insight's visualization style.
    """
    height, width = image.shape[:2]
    
    # EML-NET Principle: Multi-Layer Fusion (Object + Context)
    if saliency_map is None:
        combined_heatmap = generate_eml_heatmap_mask((height, width), boxes)
    else:
        combined_heatmap = saliency_map.copy()
    
    # --- Soften the Heatmap (Attention Insight Style) ---
    # Apply heavy Gaussian blur for soft, blobby attention pools
    # Sigma scales with image size for consistent softness
    blur_sigma = max(40, min(width, height) // 20)  # ~60-80px for 1920px width
    combined_heatmap = gaussian_filter(combined_heatmap, sigma=blur_sigma)
    
    # Normalize after blur
    if combined_heatmap.max() > 0:
        combined_heatmap = combined_heatmap / combined_heatmap.max()
    
    # Apply non-linear curve for better hot/cold distinction (push mid-values down)
    combined_heatmap = np.power(combined_heatmap, 1.5)
    
    # Apply colormap (jet: blue=cold, red=hot)
    heatmap_colored = cm.jet(combined_heatmap)[:, :, :3]
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_RGB2BGR)
    
    # Blend with original image
    alpha = 0.55  # Slightly less opacity for cleaner look
    result = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    
    return result


def generate_center_bias_map(shape: tuple) -> np.ndarray:
    """
    Generate a center-bias Gaussian map.
    Mimics the spatial prior used in DeepGaze II.
    """
    height, width = shape
    center_x, center_y = width // 2, height // 2
    
    # Create a grid of coordinates
    y, x = np.ogrid[:height, :width]
    
    # Calculate distance from center
    dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # Gaussian sigma (broad bias)
    sigma = min(width, height) * 0.4
    
    # Generate Gaussian
    bias_map = np.exp(-(dist_from_center**2) / (2 * sigma**2))
    
    return bias_map.astype(np.float32)
    
    
def generate_eml_heatmap_mask(shape: tuple, boxes: list[dict]) -> np.ndarray:
    """
    Generate the raw float heatmap mask using EML-NET principles.
    Combines Focal Layer, Structural Layer, and Center Bias.
    """
    height, width = shape
    
    # EML-NET Principle: Multi-Layer Fusion (Object + Context)
    focal_layer = np.zeros((height, width), dtype=np.float32)
    structural_layer = np.zeros((height, width), dtype=np.float32)
    
    for box in boxes:
        x, y, w, h = box['x'], box['y'], box['w'], box['h']
        b_type = box.get('type', 'focal') # Default to focal if missing
        
        if b_type == 'structural':
            # Structural/Context elements (Scene Store)
            structural_layer[y:y+h, x:x+w] += 1.0
        else:
            # Focal elements (Object Store)
            focal_layer[y:y+h, x:x+w] += 1.0
    
    # Apply Gaussian blur - Multi-scale processing
    # Focal = Fine detail (smaller sigma)
    # Structural = Coarse context (larger sigma)
    base_sigma = min(width, height) / 35
    focal_heatmap = gaussian_filter(focal_layer, sigma=base_sigma)
    structural_heatmap = gaussian_filter(structural_layer, sigma=base_sigma * 1.5)
    
    # Normalize layers independently before fusion
    if focal_heatmap.max() > 0: focal_heatmap /= focal_heatmap.max()
    if structural_heatmap.max() > 0: structural_heatmap /= structural_heatmap.max()

    # Generate Center Bias (Spatial Prior)
    center_bias = generate_center_bias_map((height, width))
        
    # EML-NET Decoder Mimicry: Weighted Fusion
    # Weights: Focal (0.6) + Structural (0.25) + CenterBias (0.15)
    combined_heatmap = (focal_heatmap * 0.6) + (structural_heatmap * 0.25) + (center_bias * 0.15)
    
    # Re-normalize final output
    if combined_heatmap.max() > 0:
        combined_heatmap = combined_heatmap / combined_heatmap.max()
        
    return combined_heatmap


def generate_center_bias_map(shape: tuple) -> np.ndarray:
    """
    Generate a center-bias Gaussian map.
    Mimics the spatial prior used in DeepGaze II.
    """
    height, width = shape
    center_x, center_y = width // 2, height // 2
    
    # Create a grid of coordinates
    y, x = np.ogrid[:height, :width]
    
    # Calculate distance from center
    dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # Gaussian sigma (broad bias)
    sigma = min(width, height) * 0.4
    
    # Generate Gaussian
    bias_map = np.exp(-(dist_from_center**2) / (2 * sigma**2))
    
    return bias_map.astype(np.float32)


def generate_aoi_image(image: np.ndarray, boxes: list[dict]) -> np.ndarray:
    """
    Generate an image with Areas of Interest (bounding boxes) overlaid.
    matches Attention Insight's 'aoi.jpg'.
    """
    result = image.copy()
    
    # Draw bounding boxes
    for box in boxes:
        x, y, w, h = box['x'], box['y'], box['w'], box['h']
        # Contrast color (green) with valid thickness
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Add label background for readability
        label = box.get('label', '')
        if len(label) > 25:
            label = label[:22] + "..."
            
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(result, (x, y - 20), (x + text_w + 4, y), (0, 255, 0), -1)
        
        cv2.putText(result, label, (x + 2, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return result


def generate_contrast_map(image: np.ndarray) -> np.ndarray:
    """
    Perceptual Contrast Map using CIE Lab color space.
    
    Computes local color distance (ΔE) which better simulates human retinal response
    to UI element "pop" compared to grayscale variance.
    
    Lab space separates:
    - L: Lightness (0-100)
    - a: Green-Red axis (-128 to +127)
    - b: Blue-Yellow axis (-128 to +127)
    
    Human perception threshold: ΔE > 2.3 is noticeable
    
    Args:
        image: BGR input image
        
    Returns:
        Perceptual contrast visualization blended with original
    """
    # Convert BGR -> Lab color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    # Split Lab channels
    L, a, b = cv2.split(lab)
    
    # Calculate local mean for each channel
    kernel_size = 15
    L_mean = cv2.blur(L, (kernel_size, kernel_size))
    a_mean = cv2.blur(a, (kernel_size, kernel_size))
    b_mean = cv2.blur(b, (kernel_size, kernel_size))
    
    # Calculate local variance for each channel
    L_sqr_mean = cv2.blur(L ** 2, (kernel_size, kernel_size))
    a_sqr_mean = cv2.blur(a ** 2, (kernel_size, kernel_size))
    b_sqr_mean = cv2.blur(b ** 2, (kernel_size, kernel_size))
    
    L_var = np.clip(L_sqr_mean - L_mean ** 2, 0, None)
    a_var = np.clip(a_sqr_mean - a_mean ** 2, 0, None)
    b_var = np.clip(b_sqr_mean - b_mean ** 2, 0, None)
    
    # Perceptual ΔE: Euclidean distance in Lab space
    # This captures both luminance AND chromatic contrast
    delta_e = np.sqrt(L_var + a_var + b_var)
    
    # Normalize to 0-1 range
    if delta_e.max() > 0:
        delta_e = delta_e / delta_e.max()
    
    # Apply perceptual threshold curve (boost visible differences)
    # Human JND (Just Noticeable Difference) is ~2.3 ΔE
    # Scale so that values above threshold become more prominent
    delta_e = np.power(delta_e, 0.7)  # Gamma correction for visibility
    
    # Apply colormap (viridis: perceptually uniform and colorblind-friendly)
    contrast_colored = cm.viridis(delta_e)[:, :, :3]
    contrast_colored = (contrast_colored * 255).astype(np.uint8)
    contrast_colored = cv2.cvtColor(contrast_colored, cv2.COLOR_RGB2BGR)
    
    # Blend with original
    alpha = 0.5
    result = cv2.addWeighted(image, 1 - alpha, contrast_colored, alpha, 0)
    
    return result


def generate_focus_map(image: np.ndarray, boxes: list[dict]) -> np.ndarray:
    """
    Create a "tunnel vision" effect by blurring peripheral areas
    while keeping attention centers sharp.
    """
    height, width = image.shape[:2]
    
    # Create a mask for sharp areas (attention centers)
    sharp_mask = np.zeros((height, width), dtype=np.float32)
    
    for box in boxes:
        x, y, w, h = box['x'], box['y'], box['w'], box['h']
        # Create elliptical sharp regions
        center_x, center_y = x + w // 2, y + h // 2
        axes = (w // 2 + 20, h // 2 + 20)
        cv2.ellipse(sharp_mask, (center_x, center_y), axes, 0, 0, 360, 1.0, -1)
    
    # Smooth the mask edges
    sharp_mask = gaussian_filter(sharp_mask, sigma=30)
    sharp_mask = np.clip(sharp_mask, 0, 1)
    
    # Create heavily blurred version
    blurred = cv2.GaussianBlur(image, (51, 51), 0)
    
    # Blend based on mask
    sharp_mask_3d = sharp_mask[:, :, np.newaxis]
    result = (image * sharp_mask_3d + blurred * (1 - sharp_mask_3d)).astype(np.uint8)
    
    return result


def calculate_focus_score(heatmap_mask: np.ndarray, boxes: list[dict], 
                          image_shape: tuple) -> float:
    """
    Calculate a composite Focus Score (0-100) based on:
    1. Attention Concentration (Attention Insight method): How 'tight' the heatmap is.
    2. UI Capture (Conversion method): How much attention falls on actionable elements.
    """
    height, width = image_shape[:2]
    total_pixels = height * width
    total_attention = heatmap_mask.sum()
    
    if total_attention == 0:
        return 50.0

    # 1. Attention Concentration (Heatmap Spread)
    # Count pixels with significant attention (> 20% of max)
    threshold = heatmap_mask.max() * 0.2
    active_pixels = np.sum(heatmap_mask > threshold)
    spread_pct = (active_pixels / total_pixels) * 100
    
    # Lower spread = Higher focus.
    # Map typical spread (5-30%) to score (100-0)
    # 5% spread -> 95 score, 30% spread -> 40 score
    concentration_score = max(0, min(100, 100 - (spread_pct * 2.5)))

    # 2. UI Capture (Efficiency)
    ui_mask = np.zeros((height, width), dtype=np.float32)
    for box in boxes:
        x, y, w, h = box['x'], box['y'], box['w'], box['h']
        ui_mask[y:y+h, x:x+w] = 1.0
    
    ui_attention = (heatmap_mask * ui_mask).sum()
    capture_score = (ui_attention / total_attention) * 100
    
    # Composite Score: 60% Concentration, 40% Capture
    # Refined based on Neurons AI methodology
    final_score = (concentration_score * 0.6) + (capture_score * 0.4)
    
    return float(min(100.0, max(0.0, final_score)))


def calculate_clarity_score(image: np.ndarray) -> float:
    """
    Calculate Visual Clarity Score using Edge Density (Feature Congestion).
    Mimics EyeQuant's 'Cleanliness' metric.
    """
    # 1. Edge Density using Canny
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    # Typical web pages have edge density 5-15%
    # We want a Bell Curve distribution:
    # - Too Empty (< 2%): Low Score
    # - Optimal (5-12%): High Score (> 80)
    # - Too Cluttered (> 20%): Low Score
    
    # Gaussian-like scoring centered at 0.08 (8% density)
    optimal_density = 0.08
    sigma = 0.06  # Width of the bell curve
    
    # Calculate Gaussian score
    score = 100 * np.exp(-((edge_density - optimal_density)**2) / (2 * sigma**2))
    
    # Boost score slightly for cleaner designs (left side of curve)
    if edge_density < optimal_density:
        score = max(score, 100 * np.exp(-((edge_density - optimal_density)**2) / (2 * (sigma*0.8)**2)))

    return float(max(10.0, min(99.0, score)))


def generate_accessibility_report(image: np.ndarray, 
                                focus_score: float = 0,
                                clarity_score: float = 0,
                                above_fold: dict = None,
                                scroll_analysis: dict = None,
                                scan_path_sequence: list = None) -> str:
    """
    WCAG 2.2 Accessibility Audit with Metric-Driven Analysis.
    
    Implements:
    1. Metric-Audit Correlation: Clarity < 60 → Visual Noise focus; Focus < 50 → CTA saliency theft
    2. Sequential Flow: Validates first 3 fixations include H1/primary navigation
    3. ATF Analysis: >50% attention drop at first fold → suggest scroll cues
    
    Args:
        image: Input image (BGR)
        focus_score: Gini coefficient-based attention concentration (0-100)
        clarity_score: Laplacian variance-based visual clarity (0-100)
        above_fold: Dict with above/below fold attention metrics
        scroll_analysis: List of scroll depth zones with attention percentages
        scan_path_sequence: Ordered list of attention boxes by predicted scanpath
        
    Returns:
        str: Markdown-formatted WCAG 2.2 accessibility audit report
    """
    # Convert OpenCV to PIL
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    
    # --- Build Predictive Scanpath Sequence String ---
    scanpath_str = "N/A"
    hierarchy_failure_flag = ""
    if scan_path_sequence and len(scan_path_sequence) > 0:
        # Extract labels for first 5 fixations
        scanpath_labels = [box.get('label', 'Unknown') for box in scan_path_sequence[:5]]
        scanpath_str = " → ".join(scanpath_labels)
        
        # Check first 3 fixations for H1 or primary navigation
        first_3_labels = [box.get('label', '').lower() for box in scan_path_sequence[:3]]
        first_3_text = ' '.join(first_3_labels)
        
        hierarchy_keywords = ['h1', 'heading', 'title', 'headline', 'nav', 'navigation', 
                             'menu', 'logo', 'brand', 'header']
        has_hierarchy = any(keyword in first_3_text for keyword in hierarchy_keywords)
        
        if not has_hierarchy:
            hierarchy_failure_flag = """
    ⚠️ **HIERARCHY FAILURE DETECTED**: The first 3 predicted fixations do NOT include the H1, 
    headline, or primary navigation. This indicates a broken visual hierarchy where 
    non-essential elements are competing for initial attention."""
    
    # --- Build Metric-Audit Correlation Directives ---
    audit_directives = []
    
    if clarity_score < 60:
        audit_directives.append(f"""
    🔴 **CLARITY ALERT** (Score: {clarity_score:.1f}%): 
    The Clarity Score is below 60%, indicating high visual noise and feature congestion.
    FOCUS YOUR AUDIT ON:
    - Reducing decorative elements that add no functional value
    - Consolidating redundant UI patterns
    - Increasing whitespace between content blocks
    - Simplifying complex visual layouts""")
    
    if focus_score < 50:
        audit_directives.append(f"""
    🔴 **FOCUS ALERT** (Score: {focus_score:.1f}%): 
    The Focus Score is below 50%, indicating saliency is scattered across non-functional elements.
    IDENTIFY WHICH ELEMENTS ARE "STEALING" ATTENTION FROM THE PRIMARY CTA:
    - Large imagery that overshadows action buttons
    - Competing secondary CTAs or promotional banners
    - Animated elements or auto-playing media
    - Dense navigation menus above the fold""")
    
    audit_directive_text = "\n".join(audit_directives) if audit_directives else "No critical metric alerts."
    
    # --- Build ATF Retention Analysis ---
    atf_analysis = ""
    retention_curve_data = ""
    
    if above_fold:
        atf_pct = above_fold.get('above_fold_attention_pct', 0)
        btf_pct = above_fold.get('below_fold_attention_pct', 0)
        
        # Calculate attention drop at first fold
        if atf_pct > 0 and btf_pct > 0:
            attention_drop = ((atf_pct - btf_pct) / atf_pct) * 100 if atf_pct > atf_pct else 0
        else:
            attention_drop = 0
            
        if atf_pct < 50:  # Attention concentration below fold
            atf_analysis = f"""
    ⚠️ **ATF ATTENTION WARNING**: Only {atf_pct:.1f}% of attention is above the fold.
    RECOMMENDATIONS:
    - Add stronger visual hooks in the hero section
    - Consider a more compelling value proposition above the fold
    - Add a scroll indicator or visual bridge to below-fold content"""
    
    # Build retention curve from scroll_analysis
    if scroll_analysis and 'zones' in scroll_analysis:
        zones = scroll_analysis['zones']
        if len(zones) >= 2:
            first_zone_attn = zones[0].get('attention_pct', 100)
            second_zone_attn = zones[1].get('attention_pct', 0) if len(zones) > 1 else 0
            
            if first_zone_attn > 0:
                drop_pct = ((first_zone_attn - second_zone_attn) / first_zone_attn) * 100
                
                if drop_pct > 50:
                    atf_analysis += f"""
    🔴 **SCROLL CLIFF DETECTED**: Attention drops by {drop_pct:.0f}% at the first fold.
    This indicates users may abandon the page before seeing key content.
    SUGGEST:
    - Add "Scroll Cues" (animated arrows, partial content previews)
    - Implement "Visual Bridges" (elements that span the fold)
    - Consider repositioning critical CTAs higher on the page"""
            
            # Format retention curve for prompt
            curve_data = [{"zone": z['name'], "attention": z['attention_pct'], 
                          "visibility": z.get('visibility_pct', 100)} for z in zones]
            retention_curve_data = f"\n    - Retention Curve Data: {json.dumps(curve_data, indent=2)}"
    
    # --- Build Complete Metrics Context ---
    metrics_context = f"""
    QUANTITATIVE DATA (Attention Analysis Engine):
    ═══════════════════════════════════════════════
    • Focus Score (Gini Coeff): {focus_score:.1f}% 
      → Measures attention concentration. <50% = scattered, >70% = focused
    • Clarity Score (Laplacian Var): {clarity_score:.1f}%
      → Measures visual cleanliness. <60% = cluttered, >75% = clean
    • Above-the-Fold Attention: {above_fold.get('above_fold_attention_pct', 0) if above_fold else 'N/A'}%
    • Predictive Scanpath: {scanpath_str}
    {retention_curve_data}
    
    ═══════════════════════════════════════════════
    AUDIT DIRECTIVES (Auto-Generated from Metrics):
    ═══════════════════════════════════════════════
    {audit_directive_text}
    {hierarchy_failure_flag}
    {atf_analysis}
    """
    
    prompt = f"""### ROLE
You are a Senior WCAG 2.2 Accessibility Auditor and UX Strategist.

### CONTEXT
You are auditing a UI based on raw screenshots and new quantitative data provided below.

{metrics_context}

### AUDIT DIRECTIVES
1. **Critical Thinking**: Do not just describe the image. Explain the *impact* of visual choices on users with cognitive disabilities, low vision, or motor impairments.
2. **Metric-Audit Correlation**: 
   - If Clarity < 60%, specifically identify which elements are creating "visual noise" (e.g., texture clashes, low-contrast text).
   - If Focus < 50%, identify which non-functional elements are "stealing" saliency from the Primary CTA.
3. **Sequential Flow**:
   - Evaluate the Predictive Scanpath. Does the H1 or primary navigation appear in the first 3 fixations? If not, explain *why* users are missing it (e.g., "The hero image faces away from the content").
4. **Cognitive Load**:
   - Assess if the layout supports "progressive disclosure" or overwhelms the user with too many choices at once.

### OUTPUT FORMAT
Generate a comprehensive WCAG 2.2 Accessibility Audit. Use EXACTLY the structure below:

## Executive Summary
Write 2-3 sentences summarizing the UI's attention health and accessibility. focus on the *why*, not just the *what*.

## Metrics Dashboard
| Metric | Value | Status |
|--------|-------|--------|
| Focus Score | {focus_score:.1f}% | [PASS] if ≥50%, [WARNING] if 35-49%, [FAIL] if <35% |
| Clarity Score | {clarity_score:.1f}% | [PASS] if ≥60%, [WARNING] if 45-59%, [FAIL] if <45% |
| ATF Attention | {above_fold.get('above_fold_attention_pct', 0) if above_fold else 'N/A'}% | [PASS] if ≥15%, [WARNING] if 8-14%, [FAIL] if <8% |

Use exactly [PASS], [WARNING], or [FAIL] in the Status column based on the thresholds above.

## Attention & Hierarchy Analysis
Analyze the scanpath data provided. Address:
- **Scanpath Flow**: Is the visual hierarchy guiding users correctly? Mention specific element ordering.
- **Cognitive Load**: Is the interface "busy" or "clean"? Does it support quick scanning?
- **CTA Visibility**: Is the primary call-to-action prominent?

## WCAG 2.2 Accessibility Audit
### Color Contrast (1.4.3, 1.4.6)
Evaluate text/background contrast. Note any potential issues with specific color combinations.

### Text Readability (1.4.4, 1.4.12)
Assess text sizing, spacing, and legibility.

### Target Size (2.5.5, 2.5.8)
Check that interactive elements are large enough for touch/click (minimum 24x24px, ideally 44x44px).

## Above-the-Fold & Scroll Analysis
Based on the retention curve data:
- How effective is the above-the-fold content at capturing attention?
- Are there clear "scent of information" cues (arrows, partial content) to encourage scrolling?

## Prioritized Recommendations
List 3-5 actionable recommendations, ordered by priority. Use professional UX terminology (e.g., "visual weight", "affordance", "proximity").
1. **Critical**: [Must fix - blocks users or violates WCAG AA]
2. **High**: [Should fix - significantly impacts UX]
3. **Medium**: [Consider fixing - improves experience]

---
IMPORTANT: Return ONLY the markdown. NO conversational filler. Start directly with ## Executive Summary."""

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[prompt, pil_image]
        )
        return response.text.strip()
    except Exception as e:
        return f"Accessibility analysis failed: {e}"


def analyze_above_fold(image: np.ndarray, boxes: list[dict], 
                       fold_y: int) -> dict:
    """
    Analyze attention distribution above vs below the fold.
    
    Args:
        image: The image to analyze
        boxes: List of attention boxes
        fold_y: The y-coordinate of the fold line (viewport height)
        
    Returns:
        Dict with above/below fold metrics
    """
    height, width = image.shape[:2]
    fold_y = min(fold_y, height)
    
    # Create attention mask
    attention_mask = np.zeros((height, width), dtype=np.float32)
    for box in boxes:
        x, y, w, h = box['x'], box['y'], box['w'], box['h']
        attention_mask[y:y+h, x:x+w] += 1.0
    
    attention_mask = gaussian_filter(attention_mask, sigma=min(width, height) / 35)
    
    # Calculate attention above and below fold
    above_fold_attention = attention_mask[:fold_y, :].sum()
    below_fold_attention = attention_mask[fold_y:, :].sum()
    total_attention = above_fold_attention + below_fold_attention
    
    if total_attention > 0:
        above_fold_pct = (above_fold_attention / total_attention) * 100
        below_fold_pct = (below_fold_attention / total_attention) * 100
    else:
        above_fold_pct = 50.0
        below_fold_pct = 50.0
    
    # Count boxes above/below fold
    boxes_above = sum(1 for b in boxes if b['y'] + b['h'] / 2 < fold_y)
    boxes_below = len(boxes) - boxes_above
    
    return {
        'fold_y': fold_y,
        'above_fold_attention_pct': float(round(above_fold_pct, 1)),
        'below_fold_attention_pct': float(round(below_fold_pct, 1)),
        'boxes_above_fold': int(boxes_above),
        'boxes_below_fold': int(boxes_below),
        'fold_ratio': float(round(fold_y / height * 100, 1)) if height > 0 else 0
    }


def generate_western_reading_prior(shape: tuple) -> np.ndarray:
    """
    Generate a "Western Reading Gravity" spatial prior mask.
    
    Implements an F-pattern bias that weights the top-left quadrant higher,
    decaying towards the bottom-right. This ensures that if two elements 
    have equal saliency, the user "reads" the top-left one first.
    
    Returns:
        np.ndarray: Spatial prior mask (0-1 float32)
    """
    height, width = shape
    
    # Create coordinate grids normalized to [0, 1]
    y_coords = np.linspace(0, 1, height, dtype=np.float32)
    x_coords = np.linspace(0, 1, width, dtype=np.float32)
    
    # Generate 2D grids
    y_grid, x_grid = np.meshgrid(y_coords, x_coords, indexing='ij')
    
    # Western reading bias: Higher weight for top-left, decaying to bottom-right
    # F-pattern formula: (1 - x)^0.3 * (1 - y)^0.5
    # x-decay is gentler (exponent 0.3) because horizontal reading is natural
    # y-decay is steeper (exponent 0.5) because vertical scrolling requires effort
    x_bias = np.power(1.0 - x_grid, 0.3)
    y_bias = np.power(1.0 - y_grid, 0.5)
    
    # Combine biases
    spatial_prior = x_bias * y_bias
    
    # Normalize to [0.5, 1.0] range to avoid over-suppression
    # This is a "soft" bias, not a hard mask
    spatial_prior = 0.5 + 0.5 * (spatial_prior / spatial_prior.max())
    
    return spatial_prior.astype(np.float32)


def generate_inhibition_kernel(shape: tuple, center: tuple, radius: int = 100) -> np.ndarray:
    """
    Generate an Inhibition of Return (IOR) kernel.
    
    Creates an inverted Gaussian mask centered at the fixation point to 
    suppress already-attended regions.
    
    Args:
        shape: (height, width) of the saliency map
        center: (x, y) coordinate of the fixation point
        radius: Radius of the inhibition zone in pixels
        
    Returns:
        np.ndarray: IOR mask (0-1 float32) where 0 = fully inhibited
    """
    height, width = shape
    cx, cy = center
    
    # Create coordinate grids
    y, x = np.ogrid[:height, :width]
    
    # Calculate Euclidean distance from center
    distance = np.sqrt((x - cx)**2 + (y - cy)**2)
    
    # Gaussian inhibition: 1 at edges (no inhibition), 0 at center (full inhibition)
    sigma = radius / 2.5  # Controls the spread of inhibition
    ior_mask = 1.0 - np.exp(-(distance**2) / (2 * sigma**2))
    
    return ior_mask.astype(np.float32)


def generate_deterministic_scanpath(saliency_map: np.ndarray, 
                                    num_fixations: int = 6,
                                    ior_radius: int = 100,
                                    apply_reading_bias: bool = True,
                                    viewport_height: int = 900,
                                    fixations_per_fold: int = 4) -> list[dict]:
    """
    Generate a deterministic scanpath using Progressive Viewport WTA with IOR.
    
    **Key Improvement**: Processes the page fold-by-fold (top-to-bottom) to simulate
    real user scrolling behavior. Users can only fixate on content within their
    current viewport, not content below the fold.
    
    Algorithm:
    1. Divide the saliency map into viewport-sized folds
    2. For each fold (top to bottom):
       a. Apply F-pattern bias within the fold
       b. Find top N saliency peaks using WTA + IOR
       c. Move to next fold (simulating scroll)
    3. IOR carries over between folds (recently seen areas stay inhibited)
    
    Args:
        saliency_map: 2D numpy array (0-1 float) from EML-NET
        num_fixations: Maximum total fixations (cap)
        ior_radius: Radius in pixels for Inhibition of Return
        apply_reading_bias: If True, apply Western F-pattern spatial prior
        viewport_height: Height of visible viewport (fold size)
        fixations_per_fold: Maximum fixations per viewport/fold
        
    Returns:
        List of fixation dicts: [{'x': int, 'y': int, 'order': int, 'duration': float, 'fold': int}]
    """
    if saliency_map is None or saliency_map.size == 0:
        return []
    
    height, width = saliency_map.shape[:2]
    
    # Ensure saliency map is float32 and normalized
    working_map = saliency_map.astype(np.float32).copy()
    if working_map.max() > 1.0:
        working_map = working_map / 255.0
    
    # --- Navigation Bar Suppression ---
    # Users don't fixate on navigation bars - they're processed peripherally.
    # Suppress the top ~100px where nav bars typically live.
    nav_zone_height = min(100, height // 10)  # 100px or 10% of image, whichever is smaller
    nav_suppression = 0.15  # Reduce nav zone saliency to 15% (still visible but won't dominate)
    working_map[:nav_zone_height, :] *= nav_suppression
    print(f"  Nav suppression: top {nav_zone_height}px reduced to {nav_suppression*100:.0f}%")
    
    # Store original map for duration calculation (before any modifications)
    original_map = working_map.copy()
    
    # Calculate number of folds
    num_folds = max(1, int(np.ceil(height / viewport_height)))
    
    fixations = []
    fixation_order = 0
    
    print(f"  Progressive Viewport: {num_folds} folds, {fixations_per_fold} fixations/fold, IOR radius: {ior_radius}px")
    
    # --- Process Each Fold Sequentially (Top to Bottom) ---
    for fold_idx in range(num_folds):
        if fixation_order >= num_fixations:
            break
            
        # Calculate fold boundaries
        fold_start_y = fold_idx * viewport_height
        fold_end_y = min((fold_idx + 1) * viewport_height, height)
        fold_height = fold_end_y - fold_start_y
        
        # Create a viewport mask (only current fold is visible)
        viewport_mask = np.zeros((height, width), dtype=np.float32)
        viewport_mask[fold_start_y:fold_end_y, :] = 1.0
        
        # Apply viewport constraint to working map
        fold_map = working_map * viewport_mask
        
        # Apply Western reading bias WITHIN this fold only
        if apply_reading_bias and fold_height > 0:
            # Create F-pattern bias for this fold (top-left of fold is priority)
            y_coords = np.linspace(0, 1, height, dtype=np.float32)
            x_coords = np.linspace(0, 1, width, dtype=np.float32)
            y_grid, x_grid = np.meshgrid(y_coords, x_coords, indexing='ij')
            
            # Within-fold Y bias: favor top of current viewport
            # Normalize y position within fold to [0, 1]
            fold_y_bias = np.ones((height, width), dtype=np.float32)
            for y in range(fold_start_y, fold_end_y):
                relative_y = (y - fold_start_y) / max(1, fold_height - 1)
                fold_y_bias[y, :] = 1.0 - (relative_y * 0.4)  # Top of fold gets 1.0, bottom gets 0.6
            
            # X bias: left-to-right reading (F-pattern)
            x_bias = np.power(1.0 - x_grid, 0.3)
            x_bias = 0.7 + 0.3 * (x_bias / x_bias.max())  # Normalize to [0.7, 1.0]
            
            reading_prior = fold_y_bias * x_bias
            fold_map = fold_map * reading_prior
        
        # --- WTA Loop for This Fold ---
        fold_fixations = 0
        while fold_fixations < fixations_per_fold and fixation_order < num_fixations:
            # Find peak in current fold
            max_val = fold_map.max()
            
            if max_val < 0.01:  # Fold exhausted
                break
            
            # Get coordinates of maximum
            max_idx = np.unravel_index(np.argmax(fold_map), fold_map.shape)
            peak_y, peak_x = int(max_idx[0]), int(max_idx[1])
            
            # Calculate fixation duration (proportional to original intensity)
            original_intensity = original_map[peak_y, peak_x]
            duration = 200 + (original_intensity * 600)  # 200-800ms
            
            fixation_order += 1
            fold_fixations += 1
            
            # Record fixation
            fixation = {
                'x': peak_x,
                'y': peak_y,
                'order': fixation_order,
                'duration': float(round(duration, 1)),
                'intensity': float(round(original_intensity, 3)),
                'fold': fold_idx + 1,
                'label': f"F{fold_idx + 1}.{fold_fixations}"
            }
            fixations.append(fixation)
            
            print(f"  Fold {fold_idx + 1} | Fix {fold_fixations}: ({peak_x}, {peak_y}) | Int: {original_intensity:.3f}")
            
            # Apply IOR to BOTH fold_map AND working_map (carries over to next folds)
            ior_mask = generate_inhibition_kernel((height, width), (peak_x, peak_y), ior_radius)
            fold_map = fold_map * ior_mask
            working_map = working_map * ior_mask  # IOR persists across folds
    
    return fixations


def scanpath_to_boxes(fixations: list[dict], box_size: int = 60) -> list[dict]:
    """
    Convert scanpath fixations to box format for compatibility with existing functions.
    
    Args:
        fixations: List from generate_deterministic_scanpath
        box_size: Size of the attention box around each fixation
        
    Returns:
        List of box dicts compatible with existing pipeline
    """
    boxes = []
    half_size = box_size // 2
    
    for fix in fixations:
        box = {
            'x': max(0, fix['x'] - half_size),
            'y': max(0, fix['y'] - half_size),
            'w': box_size,
            'h': box_size,
            'label': fix.get('label', f"Fixation {fix['order']}"),
            'order': fix['order'],
            'duration': fix.get('duration', 300),
            'type': 'focal'
        }
        boxes.append(box)
    
    return boxes


def sort_boxes_by_reading_order(boxes: list[dict]) -> list[dict]:
    """
    Sort boxes in approximate reading order (Z-pattern / F-pattern).
    Top-to-bottom, then Left-to-right.
    Uses a robust Y-banding strategy to group elements into logical rows.
    """
    if not boxes:
        return []
        
    # Create copy to avoid modifying original list
    boxes_copy = [b.copy() for b in boxes]
    
    # Calculate centroids and dynamic band height
    min_y = min(b['y'] for b in boxes_copy) if boxes_copy else 0
    max_y = max(b['y'] + b['h'] for b in boxes_copy) if boxes_copy else 1000
    total_height = max_y - min_y
    
    # Adaptive band height: 1/12th of content height (approx 12 rows per screen)
    # clamped between 40px and 120px
    band_height = max(40, min(120, total_height // 12))
    
    # Helper to get centroid
    def get_centroid(b):
        return (b['x'] + b['w'] // 2, b['y'] + b['h'] // 2)

    # Sort by: Y-Band of CENTROID -> X coordinate of CENTROID
    def sort_key(box):
        cx, cy = get_centroid(box)
        y_band = cy // band_height
        return (y_band, cx)
        
    boxes_copy.sort(key=sort_key)
        
    return boxes_copy




def generate_scanpath_visualization(image: np.ndarray, boxes: list[dict]) -> np.ndarray:
    """
    Generate a scanpath visualization connecting attention points in order.
    """
    result = image.copy()
    overlay = result.copy()
    
    # Sort boxes designed for reading flow
    sorted_boxes = sort_boxes_by_reading_order(boxes)
    
    # Draw path lines first
    points = []
    for box in sorted_boxes:
        cx = box['x'] + box['w'] // 2
        cy = box['y'] + box['h'] // 2
        points.append((cx, cy))
    
    if len(points) > 1:
        # Draw anti-aliased thick lines connecting points
        for i in range(len(points) - 1):
            pt1 = points[i]
            pt2 = points[i+1]
            cv2.line(overlay, pt1, pt2, (50, 50, 255), 4, cv2.LINE_AA)
            
    # Blend lines with some transparency
    cv2.addWeighted(overlay, 0.6, result, 0.4, 0, result)
    
    # Draw numbered circles
    for i, box in enumerate(sorted_boxes):
        cx = box['x'] + box['w'] // 2
        cy = box['y'] + box['h'] // 2
        
        # Determine radius based on box size but clamped
        radius = max(20, min(box['w'], box['h']) // 4)
        radius = min(radius, 40) # Max limit
        
        # Circle background
        cv2.circle(result, (cx, cy), radius, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(result, (cx, cy), radius, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Number
        text = str(i + 1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        text_x = cx - text_w // 2
        text_y = cy + text_h // 2
        
        cv2.putText(result, text, (text_x, text_y), font, font_scale, 
                    (255, 255, 255), thickness, cv2.LINE_AA)
                    
    return result


def generate_scroll_depth_analysis(image: np.ndarray, boxes: list[dict], 
                                   viewport_height: int = 900,
                                   saliency_map: np.ndarray = None) -> dict:
    """
    Segment the page into logical "Screens" (Folds) based on viewport height.
    """
    height, width = image.shape[:2]
    
    # Default to 900 if viewport is missing/zero
    if not viewport_height or viewport_height <= 0:
        viewport_height = 900
        
    # Calculate screen boundaries
    num_screens = int(np.ceil(height / viewport_height))
    
    if saliency_map is not None:
        attention_mask = saliency_map
    else:
        # Create attention mask from boxes (fallback)
        attention_mask = np.zeros((height, width), dtype=np.float32)
        for box in boxes:
            x, y, w, h = box['x'], box['y'], box['w'], box['h']
            attention_mask[y:y+h, x:x+w] += 1.0
            
        # Gaussian blur for smoother "attention spread"
        attention_mask = gaussian_filter(attention_mask, sigma=min(width, height) / 35)
    
    total_attention = attention_mask.sum()
    
    zone_results = []
    
    # Analyze each screen
    for i in range(num_screens):
        start_y = i * viewport_height
        end_y = min((i + 1) * viewport_height, height)
        
        # Calculate attention in this strip
        zone_attention = attention_mask[start_y:end_y, :].sum()
        zone_pct = (zone_attention / total_attention * 100) if total_attention > 0 else 0
        
        # Estimate "Visibility Opportunity" (decay)
        # Assuming 100% at top, decaying by 20% per scroll roughly
        # Formula: 100 * (0.8 ^ i)
        visibility = 100.0 * (0.8 ** i)
        
        # Count boxes in this zone
        zone_boxes = sum(1 for b in boxes 
                        if start_y <= b['y'] + b['h'] / 2 < end_y)
        
        zone_results.append({
            'name': f'Screen {i+1}',
            'label': f'Fold {i+1}',
            'start_y': start_y,
            'end_y': end_y,
            'attention_pct': float(round(zone_pct, 1)),
            'visibility_pct': float(round(visibility, 1)),
            'box_count': int(zone_boxes)
        })
    
    return {
        'zones': zone_results,
        'total_height': height,
        'viewport_height': viewport_height,
        'total_boxes': len(boxes)
    }


def draw_fold_line(image: np.ndarray, fold_y: int, 
                   above_fold_analysis: dict) -> np.ndarray:
    """
    Draw the fold line and annotations on the image.
    """
    result = image.copy()
    height, width = result.shape[:2]
    fold_y = min(fold_y, height - 1)
    
    # Draw fold line (dashed effect)
    line_color = (0, 200, 255)  # Orange/yellow
    dash_length = 20
    for x in range(0, width, dash_length * 2):
        cv2.line(result, (x, fold_y), (min(x + dash_length, width), fold_y), 
                 line_color, 3)
    
    # Add "FOLD" label
    label = f"FOLD LINE - {above_fold_analysis['above_fold_attention_pct']:.0f}% attention above"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    
    # Background for text
    (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
    cv2.rectangle(result, (10, fold_y - text_h - 15), 
                  (text_w + 20, fold_y - 5), (0, 0, 0), -1)
    cv2.putText(result, label, (15, fold_y - 10), font, font_scale, 
                line_color, thickness)
    
    return result


def generate_scroll_depth_visualization(image: np.ndarray, 
                                        scroll_analysis: dict) -> np.ndarray:
    """
    Create a visualization with scroll depth screens and retention curve.
    """
    result = image.copy()
    height, width = result.shape[:2]
    
    zones = scroll_analysis.get('zones', [])
    if not zones:
        return result
        
    # Create a sidebar for the retention graph (right 200px)
    sidebar_width = 250
    # Make sure we don't cover too much if image is small
    sidebar_width = min(sidebar_width, width // 3)
    
    # Create semi-transparent overlay for sidebar
    sidebar = result[:, width-sidebar_width:].copy()
    grey_layer = np.zeros_like(sidebar)
    grey_layer[:] = (30, 30, 30)
    # Darken right side
    cv2.addWeighted(sidebar, 0.2, grey_layer, 0.8, 0, result[:, width-sidebar_width:])
    
    # Draw retention curve points
    curve_points = []
    
    for i, zone in enumerate(zones):
        start_y = zone['start_y']
        end_y = zone['end_y']
        mid_y = int((start_y + end_y) / 2)
        
        # Draw Fold Line (unless it's the very top)
        if i > 0:
            # Dashed line logic
            line_y = start_y
            dash_len = 15
            for x in range(0, width - sidebar_width, dash_len * 2):
                 cv2.line(result, (x, line_y), (min(x + dash_len, width - sidebar_width), line_y), 
                          (200, 200, 200), 2)
            
            # Label
            cv2.putText(result, f"FOLD {i+1}", (10, line_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        # Retention Graph Point
        # X position based on visibility/attention
        # Let's use visibility_pct for the curve as it represents "Users Remaining" model
        vis_pct = zone.get('visibility_pct', 100)
        att_pct = zone.get('attention_pct', 0)
        
        graph_x = width - sidebar_width + 20 + int((sidebar_width - 40) * (vis_pct / 100))
        curve_points.append((graph_x, mid_y))
        
        # Draw Label in Sidebar
        label = f"Screen {i+1}"
        cv2.putText(result, label, (width - sidebar_width + 10, start_y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(result, f"Attn: {att_pct:.0f}%", (width - sidebar_width + 10, start_y + 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 255, 150), 1)
                   
        cv2.putText(result, f"Seen: {vis_pct:.0f}%", (width - sidebar_width + 10, start_y + 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 200, 255), 1)

    # Draw the curve connecting points
    if len(curve_points) > 1:
        # Draw smooth curve or polyline
        pts = np.array(curve_points, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(result, [pts], False, (100, 200, 255), 3, cv2.LINE_AA)
        
        # Draw dots
        for pt in curve_points:
            cv2.circle(result, pt, 5, (255, 255, 255), -1)

    return result


def run_analysis(image_path: str, output_dir: str = "output", 
                 viewport_height: int = None, device_type: str = None,
                 page_info: dict = None) -> dict:
    """
    Run the complete analysis pipeline on an image.
    
    Args:
        image_path: Path to image file
        output_dir: Directory for output files
        viewport_height: Height of viewport for above-fold analysis (optional)
        device_type: 'mobile', 'tablet', or 'desktop' (optional)
        page_info: Additional page metadata from URL capture (optional)
        
    Returns:
        Dictionary with all analysis results.
    """
    print(f"Loading image: {image_path}")
    image = load_image(image_path)
    height, width = image.shape[:2]
    
    print(f"Image size: {width}x{height}")
    
    # Detect device type if not provided
    if device_type is None:
        if width < 768:
            device_type = 'mobile'
        elif width < 1024:
            device_type = 'tablet'
        else:
            device_type = 'desktop'
    print(f"Device type: {device_type}")
    
    # Set default viewport height based on device type if not provided
    if viewport_height is None:
        viewport_heights = {
            'mobile': 667,
            'tablet': 1024,
            'desktop': 900
        }
        viewport_height = viewport_heights.get(device_type, 900)
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Generate pure saliency map FIRST
    print("Generating pure vision saliency map (EML-NET)...")
    saliency_map_uint8 = saliency_engine.predict(image)
    
    attention_mask = None
    if saliency_map_uint8 is not None:
        attention_mask = saliency_map_uint8.astype(np.float32) / 255.0
        print("  EML-NET Saliency Map generated successfully.")
    else:
        print("  Warning: EML-NET failed. Will fallback to box-based heatmap.")
    
    # Step 2: Get attention boxes from Gemini (Guided by EML-NET)
    print("Analyzing attention points with Gemini (Guided by EML-NET)...")
    boxes = get_attention_boxes(image, saliency_map=saliency_map_uint8)
    print(f"Found {len(boxes)} attention areas")
    
    # Step 3: Generate attention heatmap and AOI image
    print("Generating attention heatmap...")
    # Generate hybrid heatmap
    attention_heatmap = generate_attention_heatmap(image, boxes, saliency_map=attention_mask)
    # If we didn't have EML-NET before, we have the box-based one now inside calculate_focus_score or we need it for metrics
    if attention_mask is None:
        # Re-generate mask from boxes for metrics if EML-NET failed
        attention_mask = generate_eml_heatmap_mask((height, width), boxes)
        
    aoi_image = generate_aoi_image(image, boxes)
    
    # Note: Scanpath visualization is now generated later using deterministic WTA algorithm
    # (Step 10.5) instead of VLM-based box sorting
    
    # Step 3 (Skipped): Generate contrast map
    # print("Generating contrast map...")
    # contrast_map = generate_contrast_map(image)
    
    # Step 4 (Skipped): Generate focus map
    # print("Generating focus map...")
    # focus_map = generate_focus_map(image, boxes)
    
    # Step 5: Calculate focus score
    print("Calculating focus score...")
    focus_score = calculate_focus_score(attention_mask, boxes, image.shape)
    print(f"Focus Score: {focus_score:.1f}%")
    
    # Step 6: Above-the-fold analysis
    print("Analyzing above-the-fold content...")
    fold_y = min(viewport_height, height)
    above_fold_analysis = analyze_above_fold(image, boxes, fold_y)
    print(f"Above fold attention: {above_fold_analysis['above_fold_attention_pct']:.1f}%")
    
    # Step 7: Scroll depth analysis (Updated with viewport)
    print("Analyzing scroll depth zones...")
    scroll_analysis = generate_scroll_depth_analysis(
        image, 
        boxes, 
        viewport_height=viewport_height,
        saliency_map=attention_mask
    )
    
    # Step 8.5: Apply Fold Line to ALL images
    print("Applying fold line to all visualizations...")
    # Note: scroll_depth_image and scanpath_image get fold lines applied after generation
    image_with_fold = draw_fold_line(image.copy(), fold_y, above_fold_analysis)
    attention_with_fold = draw_fold_line(attention_heatmap, fold_y, above_fold_analysis)
    aoi_with_fold = draw_fold_line(aoi_image, fold_y, above_fold_analysis)
    
    # Step 9: Generate scroll depth visualization
    print("Generating scroll depth visualization...")
    scroll_depth_image = generate_scroll_depth_visualization(image.copy(), scroll_analysis)
    
    # Step 10: Calculate clarity score
    print("Calculating clarity score...")
    clarity_score = calculate_clarity_score(image)
    print(f"Clarity Score: {clarity_score:.1f}%")
    
    # Step 10.5: Generate Deterministic Scanpath (Progressive Viewport WTA + IOR)
    print("Generating deterministic scanpath (Progressive Viewport)...")
    
    # Calculate num_fixations: max 4 per fold/screen, capped at 50
    num_folds = max(1, int(np.ceil(height / viewport_height)))
    num_fixations = min(num_folds * 4, 50)  # Max 4 per fold, cap at 50
    
    scanpath_fixations = generate_deterministic_scanpath(
        attention_mask, 
        num_fixations=num_fixations,
        ior_radius=min(width, height) // 12,  # Smaller IOR for dense fixations
        apply_reading_bias=True,
        viewport_height=viewport_height,  # Pass viewport for fold-by-fold processing
        fixations_per_fold=4
    )
    print(f"Generated {len(scanpath_fixations)} fixations via Progressive Viewport WTA")
    
    # Convert fixations to box format for visualization
    scanpath_boxes = scanpath_to_boxes(scanpath_fixations, box_size=60)
    
    # Re-generate scanpath visualization with deterministic fixations
    scanpath_image = generate_scanpath_visualization(image, scanpath_boxes)
    scanpath_with_fold = draw_fold_line(scanpath_image, fold_y, above_fold_analysis)

    # Step 11: Generate accessibility report with scanpath sequence
    print("Generating WCAG 2.2 accessibility report...")
    accessibility_report = generate_accessibility_report(
        image,
        focus_score=focus_score,
        clarity_score=clarity_score,
        above_fold=above_fold_analysis,
        scroll_analysis=scroll_analysis,
        scan_path_sequence=scanpath_boxes  # Pass deterministic scanpath
    )
    
    # Save images
    print("Saving analysis images...")
    cv2.imwrite(os.path.join(output_dir, "original.png"), image_with_fold)
    cv2.imwrite(os.path.join(output_dir, "attention.png"), attention_with_fold)
    cv2.imwrite(os.path.join(output_dir, "aoi.png"), aoi_with_fold)
    # cv2.imwrite(os.path.join(output_dir, "contrast.png"), contrast_map)
    # cv2.imwrite(os.path.join(output_dir, "focus.png"), focus_map)
    cv2.imwrite(os.path.join(output_dir, "scanpath.png"), scanpath_with_fold)
    cv2.imwrite(os.path.join(output_dir, "fold.png"), image_with_fold) # Fold view is just original with fold
    cv2.imwrite(os.path.join(output_dir, "scroll_depth.png"), scroll_depth_image)
    
    # Convert images to base64 for HTML embedding
    def img_to_base64_data_uri(img: np.ndarray) -> str:
        _, buffer = cv2.imencode('.png', img)
        b64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/png;base64,{b64}"
    
    results = {
        'original': img_to_base64_data_uri(image_with_fold),
        'attention': img_to_base64_data_uri(attention_with_fold),
        # 'contrast': img_to_base64_data_uri(contrast_map),
        # 'focus': img_to_base64_data_uri(focus_map),
        'scanpath': img_to_base64_data_uri(scanpath_with_fold),
        'fold': img_to_base64_data_uri(image_with_fold),
        'scroll_depth': img_to_base64_data_uri(scroll_depth_image),
        'focus_score': focus_score,
        'clarity_score': clarity_score,
        'above_fold_analysis': above_fold_analysis,
        'scroll_analysis': scroll_analysis,
        'scanpath_fixations': scanpath_fixations,  # Include raw fixation data
        'accessibility_report': accessibility_report,
        'boxes': boxes,
        'image_path': os.path.basename(image_path),
        'dimensions': {'width': width, 'height': height},
        'device_type': device_type,
        'viewport_height': viewport_height,
        'page_info': page_info or {}
    }
    
    print("Analysis complete!")
    return results


if __name__ == "__main__":
    # Test with a sample image
    import sys
    if len(sys.argv) > 1:
        results = run_analysis(sys.argv[1])
        print(f"\nFocus Score: {results['focus_score']:.1f}%")
        print(f"\nAccessibility Report:\n{results['accessibility_report']}")
    else:
        print("Usage: python analyze.py <image_path_or_url>")
