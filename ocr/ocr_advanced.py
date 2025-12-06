#!/usr/bin/env python3
"""
Advanced Comic Panel Detection and Reading Order Analysis

This module provides sophisticated panel detection and text ordering for comic
book pages with complex layouts. It handles multi-panel pages and determines
optimal reading order.

Features:
    - Automatic comic panel detection using edge detection and contours
    - Panel boundary identification and isolation
    - Reading order determination (left-to-right, top-to-bottom)
    - Text block spatial analysis within panels
    - Speech bubble ordering for natural dialogue flow
    - Integration with Google Cloud Vision API

Key Classes:
    - ComicOCR: Main class for panel-aware text extraction

Algorithms:
    - Edge detection for panel boundaries
    - Contour analysis for panel regions
    - Spatial sorting for reading order
    - Bounding box analysis for text positioning

This module is used by the main OCR pipeline to handle complex comic page
layouts where simple top-to-bottom text extraction would fail.
"""
import os
import sys
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Dict

try:
    from google.cloud import vision
    import cv2
    import io
except ImportError:
    print("ERROR: Missing required packages")
    print("Install: pip install google-cloud-vision opencv-python numpy --break-system-packages")
    sys.exit(1)


class ComicOCR:
    def __init__(self):
        self.client = vision.ImageAnnotatorClient()
        
    def detect_panels(self, image_path: str) -> List[Dict]:
        """
        Detect comic panels using edge detection and contours
        """
        # Read image with OpenCV
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter for panel-sized rectangles
        height, width = img.shape[:2]
        min_area = (width * height) * 0.01  # At least 1% of image
        max_area = (width * height) * 0.9   # At most 90% of image
        
        panels = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(contour)
                # Filter for somewhat rectangular shapes
                aspect_ratio = w / h
                if 0.3 < aspect_ratio < 3.0:  # Not too wide or tall
                    panels.append({
                        'x': x,
                        'y': y,
                        'width': w,
                        'height': h,
                        'area': area,
                        'center_x': x + w/2,
                        'center_y': y + h/2
                    })
        
        # If no panels detected, treat whole image as one panel
        if not panels:
            panels = [{
                'x': 0,
                'y': 0,
                'width': width,
                'height': height,
                'area': width * height,
                'center_x': width/2,
                'center_y': height/2
            }]
        
        # Sort panels by reading order (top-to-bottom, left-to-right)
        panels = self.sort_panels_reading_order(panels)
        
        return panels
    
    def sort_panels_reading_order(self, panels: List[Dict]) -> List[Dict]:
        """
        Sort panels in typical comic reading order
        Group by rows, then left-to-right within each row
        """
        if not panels:
            return panels
        
        # Group panels into rows based on Y overlap
        rows = []
        sorted_by_y = sorted(panels, key=lambda p: p['y'])
        
        for panel in sorted_by_y:
            placed = False
            panel_top = panel['y']
            panel_bottom = panel['y'] + panel['height']
            
            for row in rows:
                # Check if this panel overlaps vertically with the row
                row_top = min(p['y'] for p in row)
                row_bottom = max(p['y'] + p['height'] for p in row)
                
                # If significant vertical overlap, add to this row
                overlap = min(panel_bottom, row_bottom) - max(panel_top, row_top)
                if overlap > panel['height'] * 0.3:  # 30% overlap threshold
                    row.append(panel)
                    placed = True
                    break
            
            if not placed:
                rows.append([panel])
        
        # Sort panels within each row by X position
        sorted_panels = []
        for row in rows:
            row_sorted = sorted(row, key=lambda p: p['x'])
            sorted_panels.extend(row_sorted)
        
        return sorted_panels
    
    def text_in_panel(self, text_block: Dict, panel: Dict) -> bool:
        """
        Check if text block is within a panel
        """
        text_x, text_y = text_block['x'], text_block['y']
        panel_x1, panel_y1 = panel['x'], panel['y']
        panel_x2, panel_y2 = panel_x1 + panel['width'], panel_y1 + panel['height']
        
        # Check if text center is within panel bounds
        return (panel_x1 <= text_x <= panel_x2 and 
                panel_y1 <= text_y <= panel_y2)
    
    def sort_text_in_panel(self, text_blocks: List[Dict]) -> List[Dict]:
        """
        Sort text within a panel (speech bubbles typically top-to-bottom)
        """
        # Group text into speech bubbles based on proximity
        bubbles = []
        used = set()
        
        for i, block in enumerate(text_blocks):
            if i in used:
                continue
            
            bubble = [block]
            used.add(i)
            
            # Find nearby text that might be in same bubble
            for j, other in enumerate(text_blocks):
                if j in used:
                    continue
                
                # Check proximity
                dist_x = abs(block['x'] - other['x'])
                dist_y = abs(block['y'] - other['y'])
                
                # If close enough, likely same bubble
                if dist_x < 50 and dist_y < 30:  # Adjust thresholds as needed
                    bubble.append(other)
                    used.add(j)
            
            bubbles.append(bubble)
        
        # Sort bubbles by position (top-to-bottom, slight left-to-right bias)
        bubbles.sort(key=lambda b: (min(t['y'] for t in b), min(t['x'] for t in b)))
        
        # Flatten bubbles back to text blocks
        sorted_blocks = []
        for bubble in bubbles:
            # Sort text within bubble
            bubble.sort(key=lambda t: (t['y'], t['x']))
            sorted_blocks.extend(bubble)
        
        return sorted_blocks
    
    def extract_text_from_comic(self, image_path: str) -> str:
        """
        Main function to extract text from comic with panel awareness
        """
        print(f"\n{'='*60}")
        print(f"ADVANCED COMIC OCR - PANEL-AWARE TEXT EXTRACTION")
        print(f"{'='*60}")
        
        print(f"Processing: {image_path}")
        
        # Detect panels
        print("Detecting panels...")
        panels = self.detect_panels(image_path)
        print(f"Found {len(panels)} panels")
        
        # Get text from Google Vision
        print("Extracting text with Google Vision...")
        try:
            with io.open(image_path, 'rb') as image_file:
                content = image_file.read()
            
            image = vision.Image(content=content)
            response = self.client.text_detection(image=image)
            
            if response.error.message:
                print(f"ERROR: {response.error.message}")
                return None
            
            texts = response.text_annotations
            if not texts:
                print("No text found!")
                return None
            
            # Build text blocks (skip first which is full text)
            text_blocks = []
            for text in texts[1:]:
                vertices = [(v.x, v.y) for v in text.bounding_poly.vertices]
                x_coords = [v[0] for v in vertices]
                y_coords = [v[1] for v in vertices]
                x, y = min(x_coords), min(y_coords)
                
                text_blocks.append({
                    'text': text.description,
                    'x': x,
                    'y': y,
                    'confidence': text.confidence if hasattr(text, 'confidence') else 1.0
                })
            
            print(f"Found {len(text_blocks)} text blocks")
            
            # Assign text to panels and sort
            all_text = []
            
            for panel_idx, panel in enumerate(panels):
                panel_text = [t for t in text_blocks if self.text_in_panel(t, panel)]
                
                if panel_text:
                    print(f"Panel {panel_idx + 1}: {len(panel_text)} text blocks")
                    
                    # Sort text within this panel
                    sorted_panel_text = self.sort_text_in_panel(panel_text)
                    
                    # Add panel separator for clarity
                    if all_text:
                        all_text.append("")  # Empty line between panels
                    
                    all_text.append(f"[Panel {panel_idx + 1}]")
                    
                    # Group text into dialogue/narration blocks
                    current_block = []
                    for text in sorted_panel_text:
                        current_block.append(text['text'])
                        
                        # Check if this might be end of a dialogue/thought
                        if any(text['text'].endswith(p) for p in ['.', '!', '?', '...', '--']):
                            if current_block:
                                all_text.append(' '.join(current_block))
                                current_block = []
                    
                    # Add any remaining text
                    if current_block:
                        all_text.append(' '.join(current_block))
            
            return '\n'.join(all_text)
            
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def enhance_with_preprocessing(self, image_path: str) -> str:
        """
        Preprocess image to improve OCR accuracy
        """
        print("Applying image preprocessing...")
        
        # Read image
        img = cv2.imread(image_path)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to get binary image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Denoise
        denoised = cv2.medianBlur(binary, 1)
        
        # Save preprocessed image
        temp_path = "temp_preprocessed.png"
        cv2.imwrite(temp_path, denoised)
        
        # Run OCR on preprocessed image
        result = self.extract_text_from_comic(temp_path)
        
        # Clean up
        os.remove(temp_path)
        
        return result


def main():
    print("="*60)
    print("ADVANCED COMIC OCR WITH PANEL DETECTION")
    print("="*60)
    
    # Create output directory
    output_dir = Path("ocr_outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Check for credentials
    if 'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ:
        cred_files = list(Path('.').glob('*credentials*.json'))
        if cred_files:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(cred_files[0])
            print(f"Using credentials: {cred_files[0]}")
        else:
            print("ERROR: No Google Cloud credentials found!")
            print("Please set GOOGLE_APPLICATION_CREDENTIALS environment variable")
            return 1
    
    # Get image path
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Look for comic images
        images = []
        for ext in ['.jpg', '.jpeg', '.png', '.webp']:
            images.extend(list(Path('.').glob(f'*{ext}')))
            images.extend(list(Path('.').glob(f'*{ext.upper()}')))
        
        if not images:
            print("No images found!")
            print("\nUsage: python comic_ocr_advanced.py <comic_image>")
            return 1
        
        print("\nFound images:")
        for i, img in enumerate(images):
            print(f"  {i+1}. {img}")
        
        choice = input("\nSelect image number (or press Enter for first): ").strip()
        if choice and choice.isdigit():
            image_path = str(images[int(choice) - 1])
        else:
            image_path = str(images[0])
    
    if not os.path.exists(image_path):
        print(f"ERROR: Image not found: {image_path}")
        return 1
    
    # Create OCR instance
    ocr = ComicOCR()
    
    # Extract text
    extracted_text = ocr.extract_text_from_comic(image_path)
    
    if extracted_text:
        print("\n" + "="*60)
        print("EXTRACTED TEXT (PANEL-AWARE ORDERING)")
        print("="*60)
        print(extracted_text)
        print("="*60)
        
        # Save to file
        output_file = output_dir / "comic_text_extracted.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(extracted_text)
        
        print(f"\n✓ Text saved to: {output_file}")
        
        # Ask about preprocessing
        enhance = input("\nTry enhanced preprocessing for better accuracy? (y/n): ").strip().lower()
        if enhance == 'y':
            enhanced_text = ocr.enhance_with_preprocessing(image_path)
            if enhanced_text:
                enhanced_file = output_dir / "comic_text_enhanced.txt"
                with open(enhanced_file, 'w', encoding='utf-8') as f:
                    f.write(enhanced_text)
                print(f"✓ Enhanced text saved to: {enhanced_file}")
        
        return 0
    else:
        print("\n✗ Failed to extract text")
        return 1


if __name__ == "__main__":
    exit(main())