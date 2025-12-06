#!/usr/bin/env python3
"""
Speech Bubble Detection and Text Extraction Utility

This module provides specialized tools for detecting and extracting text from
speech bubbles in comic book images. It combines computer vision techniques
with OCR to identify dialogue regions.

Features:
    - Speech bubble contour detection using OpenCV
    - Bubble region isolation and preprocessing
    - Text extraction from bubble regions using Tesseract OCR
    - Confidence scoring for detected text
    - Integration with Google Cloud Vision API for enhanced accuracy

Key Classes:
    - SpeechBubbleDetector: Main class for bubble detection and text extraction

Use Cases:
    - Extracting dialogue from traditional comic panels
    - Detecting speech bubble locations for layout analysis
    - Preprocessing bubble regions for better OCR accuracy

This is a utility module used by the main OCR pipeline (narration/vision_ocr.py)
to improve text extraction quality specifically for comic book dialogue.
"""
import os
import sys
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple

try:
    from google.cloud import vision
    import cv2
    import numpy as np
    import pytesseract
    from PIL import Image
except ImportError:
    print("ERROR: Missing required packages")
    print("Install with:")
    print("  pip install google-cloud-vision opencv-python numpy pillow pytesseract --break-system-packages")
    print("  sudo apt-get install tesseract-ocr")
    sys.exit(1)


class SpeechBubbleDetector:
    def __init__(self):
        self.vision_client = None
        try:
            self.vision_client = vision.ImageAnnotatorClient()
        except:
            print("Warning: Google Vision API not configured, using Tesseract only")
    
    def detect_bubbles(self, image_path: str) -> List[Dict]:
        """
        Detect speech bubbles using contour detection
        """
        img = cv2.imread(image_path)
        height, width = img.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to isolate white/light areas (typical bubbles)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bubbles = []
        min_area = width * height * 0.001  # Minimum bubble size
        max_area = width * height * 0.3    # Maximum bubble size
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check if shape is bubble-like (not too elongated)
                aspect_ratio = w / h
                if 0.4 < aspect_ratio < 2.5:
                    # Check circularity
                    perimeter = cv2.arcLength(contour, True)
                    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                    
                    # Bubbles tend to be somewhat circular
                    if circularity > 0.3:  # Adjust threshold as needed
                        bubbles.append({
                            'x': x,
                            'y': y,
                            'width': w,
                            'height': h,
                            'area': area,
                            'circularity': circularity,
                            'contour': contour
                        })
        
        return bubbles
    
    def extract_text_from_region(self, img, x, y, w, h, method='both'):
        """
        Extract text from a specific region using OCR
        """
        # Crop region
        roi = img[y:y+h, x:x+w]
        
        # Preprocess for better OCR
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        
        # Increase contrast
        enhanced = cv2.convertScaleAbs(gray, alpha=1.5, beta=10)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced)
        
        text_results = []
        
        # Method 1: Tesseract OCR
        if method in ['tesseract', 'both']:
            try:
                # Convert to PIL Image
                pil_image = Image.fromarray(denoised)
                
                # Use multiple PSM modes for comics
                psm_modes = [6, 11, 8]  # Uniform block, sparse text, single word
                
                for psm in psm_modes:
                    config = f'--psm {psm} --oem 3'
                    text = pytesseract.image_to_string(pil_image, config=config).strip()
                    if text:
                        text_results.append(('tesseract', text, psm))
            except Exception as e:
                print(f"Tesseract error: {e}")
        
        # Method 2: Google Vision API
        if method in ['vision', 'both'] and self.vision_client:
            try:
                # Convert region to bytes
                _, buffer = cv2.imencode('.png', roi)
                image = vision.Image(content=buffer.tobytes())
                
                response = self.vision_client.text_detection(image=image)
                texts = response.text_annotations
                
                if texts:
                    # First annotation contains all text
                    text_results.append(('vision', texts[0].description.strip(), None))
            except Exception as e:
                print(f"Vision API error: {e}")
        
        # Choose best result
        if text_results:
            # Prefer Vision API if available and has content
            vision_results = [r for r in text_results if r[0] == 'vision' and r[1]]
            if vision_results:
                return vision_results[0][1]
            
            # Otherwise use longest Tesseract result
            tesseract_results = [r for r in text_results if r[0] == 'tesseract' and r[1]]
            if tesseract_results:
                return max(tesseract_results, key=lambda x: len(x[1]))[1]
        
        return ""
    
    def process_comic_page(self, image_path: str) -> Dict:
        """
        Process entire comic page
        """
        print(f"\nProcessing: {image_path}")
        
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Cannot read image {image_path}")
            return None
        
        height, width = img.shape[:2]
        print(f"Image size: {width}x{height}")
        
        # Method 1: Detect speech bubbles
        print("Detecting speech bubbles...")
        bubbles = self.detect_bubbles(image_path)
        print(f"Found {len(bubbles)} potential speech bubbles")
        
        # Sort bubbles by reading order
        bubbles = sorted(bubbles, key=lambda b: (b['y'] // 100, b['x']))
        
        # Extract text from each bubble
        bubble_texts = []
        for i, bubble in enumerate(bubbles):
            text = self.extract_text_from_region(
                img, 
                bubble['x'], bubble['y'], 
                bubble['width'], bubble['height']
            )
            if text:
                bubble_texts.append({
                    'bubble_id': i + 1,
                    'text': text,
                    'position': (bubble['x'], bubble['y']),
                    'size': (bubble['width'], bubble['height'])
                })
                print(f"Bubble {i+1}: {text[:50]}...")
        
        # Method 2: Also do full page OCR for comparison
        print("\nPerforming full page OCR...")
        full_text = self.extract_text_from_region(img, 0, 0, width, height)
        
        return {
            'bubble_texts': bubble_texts,
            'full_text': full_text,
            'image_size': (width, height),
            'num_bubbles': len(bubble_texts)
        }
    
    def smart_text_ordering(self, results: Dict) -> str:
        """
        Smart ordering of extracted text based on comic conventions
        """
        if not results or not results.get('bubble_texts'):
            return results.get('full_text', '') if results else ''
        
        # Group bubbles into panels (based on vertical position)
        panels = []
        current_panel = []
        last_y = -1
        panel_threshold = 50  # Pixels between panels
        
        for bubble in results['bubble_texts']:
            y = bubble['position'][1]
            if last_y == -1 or abs(y - last_y) < panel_threshold:
                current_panel.append(bubble)
            else:
                if current_panel:
                    panels.append(current_panel)
                current_panel = [bubble]
            last_y = y
        
        if current_panel:
            panels.append(current_panel)
        
        # Build ordered text
        ordered_lines = []
        for panel_idx, panel in enumerate(panels):
            # Sort bubbles in panel left-to-right
            panel_sorted = sorted(panel, key=lambda b: b['position'][0])
            
            if panel_sorted:
                ordered_lines.append(f"\n[Panel {panel_idx + 1}]")
                for bubble in panel_sorted:
                    # Clean up text
                    text = bubble['text']
                    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
                    text = text.strip()
                    if text:
                        ordered_lines.append(text)
        
        return '\n'.join(ordered_lines)


def create_debug_image(image_path: str, bubbles: List[Dict], output_path: str = "debug_bubbles.jpg"):
    """
    Create debug image showing detected bubbles
    """
    img = cv2.imread(image_path)
    
    # Draw rectangles around detected bubbles
    for i, bubble in enumerate(bubbles):
        x, y, w, h = bubble['x'], bubble['y'], bubble['width'], bubble['height']
        
        # Draw rectangle
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Add label
        cv2.putText(img, f"B{i+1}", (x, y-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imwrite(output_path, img)
    print(f"Debug image saved to: {output_path}")


def main():
    print("="*60)
    print("COMIC SPEECH BUBBLE TEXT EXTRACTOR")
    print("="*60)
    
    # Create output directory
    output_dir = Path("ocr_outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Check for Tesseract
    try:
        pytesseract.get_tesseract_version()
        print("✓ Tesseract OCR found")
    except:
        print("⚠ Tesseract not found. Install with: sudo apt-get install tesseract-ocr")
    
    # Check for Google Vision credentials
    if 'GOOGLE_APPLICATION_CREDENTIALS' in os.environ:
        print("✓ Google Vision API configured")
    else:
        cred_files = list(Path('.').glob('*credentials*.json'))
        if cred_files:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(cred_files[0])
            print(f"✓ Using credentials: {cred_files[0]}")
        else:
            print("⚠ No Google Cloud credentials found (will use Tesseract only)")
    
    # Get image
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Find comic images
        images = []
        for ext in ['.jpg', '.jpeg', '.png', '.webp']:
            images.extend(list(Path('.').glob(f'*{ext}')))
            images.extend(list(Path('.').glob(f'*{ext.upper()}')))
        
        # Filter out debug images
        images = [img for img in images if 'debug' not in str(img).lower()]
        
        if not images:
            print("\nNo images found!")
            print("Usage: python bubble_extractor.py <comic_image>")
            return 1
        
        print(f"\nFound {len(images)} images:")
        for i, img in enumerate(images[:10]):  # Show max 10
            print(f"  {i+1}. {img.name}")
        
        choice = input("\nSelect image number: ").strip()
        try:
            image_path = str(images[int(choice) - 1])
        except:
            image_path = str(images[0])
    
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found")
        return 1
    
    # Create detector
    detector = SpeechBubbleDetector()
    
    # Process comic
    results = detector.process_comic_page(image_path)
    
    if results:
        # Generate ordered text
        ordered_text = detector.smart_text_ordering(results)
        
        print("\n" + "="*60)
        print("EXTRACTED TEXT (BUBBLE-AWARE ORDERING)")
        print("="*60)
        print(ordered_text)
        print("="*60)
        
        # Save results
        output_file = output_dir / "comic_bubbles_text.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(ordered_text)
        print(f"\n✓ Ordered text saved to: {output_file}")
        
        # Save detailed results
        json_file = output_dir / "comic_bubbles_data.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            # Convert to serializable format
            save_data = {
                'bubble_texts': results['bubble_texts'],
                'full_text': results['full_text'],
                'num_bubbles': results['num_bubbles']
            }
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        print(f"✓ Detailed data saved to: {json_file}")
        
        # Ask about debug image
        debug = input("\nCreate debug image showing detected bubbles? (y/n): ").strip().lower()
        if debug == 'y':
            # Recreate bubbles for debug
            bubbles = detector.detect_bubbles(image_path)
            create_debug_image(image_path, bubbles, str(output_dir / "debug_bubbles.jpg"))
        
        print("\n✓ Processing complete!")
        return 0
    else:
        print("\n✗ Failed to process comic")
        return 1


if __name__ == "__main__":
    exit(main())