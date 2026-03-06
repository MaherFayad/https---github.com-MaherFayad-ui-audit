"""
Local UI Analyzer - Flask API Server
Serves the analysis pipeline via REST API for the React frontend.
"""

import os
import sys
import json
import uuid
import time
import base64
import traceback
from pathlib import Path
from io import BytesIO

import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

from analyze import run_analysis, generate_accessibility_report
from screenshot import capture_website, is_url, get_viewport_config

# Gemini imports for UX Overview
from dotenv import load_dotenv
load_dotenv()
from google import genai

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY not found in .env file. UX Overview will fail.")

GEMINI_MODEL = "gemini-3.1-pro-preview"
gemini_client = None
if GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)

app = Flask(__name__, static_folder='frontend/dist', static_url_path='')
CORS(app)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


@app.route('/')
def serve_frontend():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """
    Run the full analysis pipeline.
    
    Accepts JSON body:
    {
        "url": "https://...",           # OR upload via multipart
        "device_type": "desktop",       # mobile | tablet | desktop
        "viewport_width": 1920,         # custom width (desktop)
        "context": "This is a landing page for..."  # optional page context
    }
    """
    try:
        # Parse input
        if request.content_type and 'multipart' in request.content_type:
            url = request.form.get('url', '')
            device_type = request.form.get('device_type', 'desktop')
            viewport_width = int(request.form.get('viewport_width', 1920))
            context = request.form.get('context', '')
            uploaded_file = request.files.get('image')
        else:
            data = request.get_json() or {}
            url = data.get('url', '')
            device_type = data.get('device_type', 'desktop')
            viewport_width = data.get('viewport_width', 1920)
            context = data.get('context', '')
            uploaded_file = None

        # Create unique output directory for this analysis
        analysis_id = str(uuid.uuid4())[:8]
        analysis_dir = OUTPUT_DIR / analysis_id
        analysis_dir.mkdir(exist_ok=True)

        page_info = None
        image_path = None
        viewport_height = None

        # Viewport config
        viewport_heights = {'mobile': 667, 'tablet': 1024, 'desktop': 900}
        
        if uploaded_file:
            # Handle file upload
            filename = secure_filename(uploaded_file.filename or 'upload.png')
            image_path = str(analysis_dir / filename)
            uploaded_file.save(image_path)
            viewport_height = viewport_heights.get(device_type, 900)
            
        elif url and is_url(url):
            # Handle URL capture
            print(f"Capturing website: {url}")
            
            if device_type == 'desktop':
                custom_viewport = (viewport_width, 1080)
                viewport_preset = 'desktop'
            elif device_type == 'mobile':
                custom_viewport = None
                viewport_preset = 'mobile'
            elif device_type == 'tablet':
                custom_viewport = None
                viewport_preset = 'tablet'
            else:
                custom_viewport = (viewport_width, 1080)
                viewport_preset = 'desktop'
            
            screenshot_path = str(analysis_dir / "screenshot.png")
            page_info = capture_website(
                url,
                viewport=viewport_preset,
                custom_size=custom_viewport,
                output_path=screenshot_path,
                wait_time=2000
            )
            
            image_path = page_info['fullpage_screenshot']
            viewport_height = page_info['viewport_height']
            device_type = page_info['device_type']
        else:
            return jsonify({'error': 'Please provide a URL or upload an image'}), 400

        # Run analysis
        print(f"Running analysis on: {image_path}")
        results = run_analysis(
            image_path,
            str(analysis_dir),
            viewport_height=viewport_height,
            device_type=device_type,
            page_info=page_info
        )

        # Generate UX Overview with Gemini if context is provided
        ux_overview = None
        if context.strip():
            ux_overview = generate_ux_overview(results, context, image_path)
            results['ux_overview'] = ux_overview

        # Build response
        response = {
            'analysis_id': analysis_id,
            'original': results.get('original'),
            'attention': results.get('attention'),
            'scanpath': results.get('scanpath'),
            'scroll_depth': results.get('scroll_depth'),
            'mouse_movement': results.get('mouse_movement'),
            'focus_score': float(results.get('focus_score', 0)),
            'clarity_score': float(results.get('clarity_score', 0)),
            'above_fold_analysis': results.get('above_fold_analysis'),
            'scroll_analysis': results.get('scroll_analysis'),
            'scanpath_fixations': results.get('scanpath_fixations'),
            'accessibility_report': results.get('accessibility_report'),
            'ux_overview': ux_overview,
            'boxes': results.get('boxes', []),
            'dimensions': results.get('dimensions'),
            'device_type': device_type,
            'viewport_height': viewport_height,
            'page_info': results.get('page_info', {}),
        }

        return app.response_class(
            response=json.dumps(response, cls=NumpyEncoder),
            status=200,
            mimetype='application/json'
        )
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


def generate_ux_overview(results: dict, context: str, image_path: str) -> str:
    """
    Generate a comprehensive UX overview using Gemini 3.1 Pro Preview.
    Takes the analysis results + user-provided context to give actionable UX advice.
    """
    try:
        # Smart Adaptive Slicing for Gemini Vision
        from PIL import Image
        import io
        
        full_img = Image.open(image_path)
        original_width, original_height = full_img.size
        
        # 1. Standardize Width: 1280px is optimal for Gemini vision (HD)
        # Higher widths (1920+) often trigger dimension errors on very long pages.
        TARGET_WIDTH = 1280
        scale_ratio = TARGET_WIDTH / float(original_width)
        scaled_height = int(original_height * scale_ratio)
        
        print(f"  Gemini: Processing {original_width}x{original_height} page...")
        print(f"  Gemini: Standardizing to {TARGET_WIDTH}px width (Total height: {scaled_height}px)")
        
        # Resize full image to standardized width
        resized_full_img = full_img.resize((TARGET_WIDTH, scaled_height), Image.Resampling.LANCZOS)
        
        # 2. Slice into manageable Vision Chunks
        # Limit to max 5 chunks to prevent request timeout/payload issues
        MAX_CHUNKS = 5
        CHUNK_HEIGHT = 1800 # ~1.5x viewport height
        OVERLAP = 150
        
        # If the page is extremely long, we increase chunk height to keep count under MAX_CHUNKS
        effective_chunk_height = max(CHUNK_HEIGHT, (scaled_height // MAX_CHUNKS) + 100)
        
        img_parts = []
        y_start = 0
        while y_start < scaled_height and len(img_parts) < MAX_CHUNKS:
            y_end = min(y_start + effective_chunk_height, scaled_height)
            
            # Crop the chunk
            chunk_img = resized_full_img.crop((0, y_start, TARGET_WIDTH, y_end))
            
            # Ensure RGB
            if chunk_img.mode in ("RGBA", "P"):
                chunk_img = chunk_img.convert("RGB")
            
            img_parts.append(chunk_img)
            
            if y_end == scaled_height:
                break
            y_start = y_end - OVERLAP

        print(f"  Gemini: Sending {len(img_parts)} high-res chunks for analysis...")
        
        focus_score = results.get('focus_score', 0)
        clarity_score = results.get('clarity_score', 0)
        num_elements = len(results.get('boxes', []))
        above_fold = results.get('above_fold_analysis', {})
        
        prompt = f"""You are a Senior UX/UI Strategist. I am providing you with a **sequential scroll journey** of a single webpage.
        
### HOW TO ANALYZE THIS INPUT:
- **Visual Sequence**: I have provided {len(img_parts)} images. Image 1 is the Hero/Landing (Top), and they proceed as you scroll down.
- **Continuous Logic**: Treat this as a single continuous experience. Ignore the ~150px overlaps at the top/bottom of images; they are just seams.
- **Goal**: Evaluate how effectively the page tells a story and leads the user to take action as they move through the sections.

### Page Metrics:
- **Focus Score**: {focus_score:.1f}% (concentration on key elements)
- **Visual Clarity**: {clarity_score:.1f}% (cleanliness vs clutter)
- **Viewport Height**: {above_fold.get('fold_y', 'N/A')}px
- **Context**: {context}

### Your Task:
Provide a comprehensive UX audit. Use this exact structure:

## Executive Summary
Summarize the page's design effectiveness and conversion potential in 3 sentences.

## 1. Above the Fold (Slice #1)
Analyze the value proposition and the immediate "Hero" impact. Is the CTA obvious?

## 2. Scroll Journey & Engagement
How does the narrative evolve across Slices #2 to #{len(img_parts)}? Is there a "scent of information" pulling the user down?

## 3. Visual Consistency & Brand Trust
Does the design feel cohesive? Is typography and color used effectively?

## 4. Top 5 Actionable Recommendations
List 5 specific, prioritized changes to improve conversion and UX.

Format in clean markdown. Start directly with ## Executive Summary."""

        # Call Gemini using the GenAI SDK's native list support [text, img1, img2, ...]
        contents = [prompt] + img_parts
        
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents
        )
        
        return response.text
        
    except Exception as e:
        print(f"UX Overview generation failed: {e}")
        return f"*UX Overview could not be generated: {e}*"


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': 'EML-NET v2 + OmniParser V2'})


if __name__ == '__main__':
    print("=" * 60)
    print("  🎯 Local UI Analyzer - API Server")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=False)
