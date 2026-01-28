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
    Merge overlapping boxes from different chunks.
    Uses IoU (Intersection over Union) to deduplicate.
    """
    if not all_boxes:
        return []
    
    def calc_iou(box1, box2):
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
    
    # Sort by area (larger first)
    sorted_boxes = sorted(all_boxes, key=lambda b: b['w'] * b['h'], reverse=True)
    merged = []
    
    for box in sorted_boxes:
        is_duplicate = False
        for existing in merged:
            if calc_iou(box, existing) > iou_threshold:
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
    """
    height, width = image.shape[:2]
    
    # EML-NET Principle: Multi-Layer Fusion (Object + Context)
    if saliency_map is None:
        combined_heatmap = generate_eml_heatmap_mask((height, width), boxes)
    else:
        combined_heatmap = saliency_map
    
    # Apply colormap (jet: blue=cold, red=hot)
    heatmap_colored = cm.jet(combined_heatmap)[:, :, :3]
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_RGB2BGR)
    
    # Blend with original image
    alpha = 0.6
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
    Compute local pixel variance to highlight areas of high visual complexity.
    Returns a visualization of the contrast/complexity map.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    # Calculate local variance using a sliding window
    kernel_size = 15
    mean = cv2.blur(gray, (kernel_size, kernel_size))
    sqr_mean = cv2.blur(gray ** 2, (kernel_size, kernel_size))
    variance = sqr_mean - mean ** 2
    
    # Normalize variance
    variance = np.clip(variance, 0, None)
    if variance.max() > 0:
        variance = variance / variance.max()
    
    # Apply colormap (viridis for contrast)
    variance_colored = cm.viridis(variance)[:, :, :3]
    variance_colored = (variance_colored * 255).astype(np.uint8)
    variance_colored = cv2.cvtColor(variance_colored, cv2.COLOR_RGB2BGR)
    
    # Blend with original
    alpha = 0.5
    result = cv2.addWeighted(image, 1 - alpha, variance_colored, alpha, 0)
    
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
                                scroll_analysis: list = None) -> str:
    """
    Query Gemini to generate a comprehensive predictive UX/UI report
    integrating visual attention data with accessibility standards.
    """
    # Convert OpenCV to PIL
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    
    # Format metrics for the prompt
    metrics_context = f"""
    QUANTITATIVE DATA (from Attention Analysis):
    - Focus Score: {focus_score:.1f}% (Heatmap Concentration + UI Capture. >50% is good)
    - Clarity Score: {clarity_score:.1f}% (Visual Cleanliness/Low Clutter. >60% is good)
    - Above the Fold Attention: {above_fold['above_fold_attention_pct'] if above_fold else 0}%
    - Attention by Scroll Depth: {json.dumps(scroll_analysis, indent=2) if scroll_analysis else 'N/A'}
    """
    
    prompt = f"""Act as a Senior UX Researcher & Cognitive Scientist.
Analyze this UI screenshot and the provided attention metrics to generate a "Predictive Attention & Accessibility Report".

{metrics_context}

Evaluate the design based on:
1. **Cognitive Load & Clarity**: 
   - Does the Clarity Score ({clarity_score:.1f}%) suggest the design is clean or cluttered? 
   - Does the Focus Score ({focus_score:.1f}%) show that users are finding key elements efficiently?
2. **Visual Hierarchy & Scroll**: detailed critique of how attention flows down the page. Is the above-fold attention adequate?
3. **Accessibility & Contrast**: Identify specific WCAG failures (colors, font sizes).
4. **Conversion Optimization**: Are the key CTAs receiving attention?

Provide a professional, actionable report with these sections:
- **Executive Summary**
- **Attention & Clarity Analysis** (Cite the data)
- **Accessibility & Clarity**
- **Recommendations**

Format with Markdown. 
IMPORTANT: Return ONLY the markdown content. Do NOT include any conversational filler like "Here is the report" or "As an AI". Start directly with the # Report Title or ## Section.
"""

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
    # Use 10% of image height or 50px as default band height if passed
    # but we don't have image height here easily unless we inspect boxes or pass it in.
    # Let's infer approximate scale from the boxes range.
    min_y = min(b['y'] for b in boxes_copy) if boxes_copy else 0
    max_y = max(b['y'] + b['h'] for b in boxes_copy) if boxes_copy else 1000
    total_height = max_y - min_y
    
    # Adaptive band height: 1/12th of content height (approx 12 rows per screen)
    # clamped between 40px and 120px
    band_height = max(40, min(120, total_height // 12))
    
    # Helper to get centroid
    def get_centroid(b):
        return (b['x'] + b['w'] // 2, b['y'] + b['h'] // 2)

    # We sort by:
    # 1. Y-Band of CENTROID -> Robust row grouping
    # 2. X coordinate of CENTROID -> Left to Right reading
    
    def sort_key(box):
        cx, cy = get_centroid(box)
        y_band = cy // band_height
        return (y_band, cx)
        
    boxes_copy.sort(key=sort_key)
    
    # Debug print to understand sorting
    print(f"DEBUG: Sorting boxes with band_height={band_height}")
    for i, b in enumerate(boxes_copy):
        cx, cy = get_centroid(b)
        print(f"  Box {i+1}: cy={cy} (Band {cy // band_height}), cx={cx} -> Label: {b.get('label', '?')}")
        
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


def generate_scroll_depth_analysis(image: np.ndarray, boxes: list[dict], viewport_height: int = 900) -> dict:
    """
    Segment the page into logical "Screens" (Folds) based on viewport height.
    """
    height, width = image.shape[:2]
    
    # Default to 900 if viewport is missing/zero
    if not viewport_height or viewport_height <= 0:
        viewport_height = 900
        
    # Calculate screen boundaries
    num_screens = int(np.ceil(height / viewport_height))
    
    # Create attention mask for exact calculation
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
    
    # Step 3.5: Generate Scanpath (New)
    print("Generating scanpath visualization...")
    scanpath_image = generate_scanpath_visualization(image, boxes)
    
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
    scroll_analysis = generate_scroll_depth_analysis(image, boxes, viewport_height=viewport_height)
    
    # Step 8.5: Apply Fold Line to ALL images
    print("Applying fold line to all visualizations...")
    # Note: scroll_depth_image gets its own special folds in generation, so we skip it here to avoid double drawing
    image_with_fold = draw_fold_line(image.copy(), fold_y, above_fold_analysis)
    attention_with_fold = draw_fold_line(attention_heatmap, fold_y, above_fold_analysis)
    aoi_with_fold = draw_fold_line(aoi_image, fold_y, above_fold_analysis)
    scanpath_with_fold = draw_fold_line(scanpath_image, fold_y, above_fold_analysis)
    
    # Step 9: Generate scroll depth visualization
    print("Generating scroll depth visualization...")
    scroll_depth_image = generate_scroll_depth_visualization(image.copy(), scroll_analysis)
    
    # Step 10: Calculate clarity score
    print("Calculating clarity score...")
    clarity_score = calculate_clarity_score(image)
    print(f"Clarity Score: {clarity_score:.1f}%")

    # Step 11: Generate accessibility report
    print("Generating accessibility report...")
    accessibility_report = generate_accessibility_report(image,
                                                      focus_score=focus_score,
                                                      clarity_score=clarity_score,
                                                      above_fold=above_fold_analysis,
                                                      scroll_analysis=scroll_analysis)
    
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
        'above_fold_analysis': above_fold_analysis,
        'scroll_analysis': scroll_analysis,
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
