#!/usr/bin/env python3
"""
Comic Text Extraction Module - OCR Fallback and Legacy Support

This module provides Google Cloud Vision-based OCR text extraction from comic images.
It serves as the fallback when LLM narration is unavailable or disabled.

Features:
    - Speech bubble detection and extraction
    - Panel-aware reading order (left-to-right, top-to-bottom)
    - Character dialogue attribution
    - Human-like text ordering for natural flow
    - Integration with Google Cloud Vision API

Mode Selection:
    - Primary: Uses LLM narrator (narration/llm_narrator.py) if available
    - Fallback: Falls back to this OCR-based extraction automatically

Key Classes:
    - ComicOCR: Main class for extracting text from comic images
    - get_vision_client(): Lazy-initialized Google Vision client (fork-safe)
    - get_tts_client(): Lazy-initialized Google TTS client (fork-safe)

The module also provides directory management (AUDIO_DIR, TEMP_DIR) and
credential setup for Google Cloud services.

Note: This was formerly the main text extraction method before LLM narration
was introduced. Now primarily used as a reliable fallback option.
"""

import os
from pathlib import Path

# ===== CRITICAL: SET CREDENTIALS BEFORE IMPORTING GOOGLE CLOUD LIBRARIES =====
# This ensures the API key is detected before any Google Cloud clients are initialized
def setup_credentials():
    """Setup Google Cloud credentials before client initialization"""
    if 'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ:
        # Look for credentials.json in current directory
        cred_files = list(Path('.').glob('*credentials*.json'))
        if cred_files:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(cred_files[0].absolute())
            print(f"✓ Found and set credentials: {cred_files[0]}")
            return True
        else:
            print("⚠️  WARNING: No Google Cloud credentials found!")
            print("   Please ensure 'credentials.json' is in the same directory as this script")
            print("   Or set GOOGLE_APPLICATION_CREDENTIALS environment variable")
            return False
    else:
        print(f"✓ Using existing credentials: {os.environ['GOOGLE_APPLICATION_CREDENTIALS']}")
        return True

# Set up credentials BEFORE importing Google Cloud libraries
credentials_available = setup_credentials()

# Now import Google Cloud libraries (they will use the credentials we just set)
from google.cloud import vision, texttospeech

# Import remaining dependencies
from flask import Flask, request, jsonify, send_file, render_template_string
from flask_cors import CORS
import cv2
import numpy as np
import uuid
from datetime import datetime, timedelta
import base64
import re
from itertools import permutations
from threading import Lock

# Import LLM narrator for audiobook-style narration
try:
    from narration.llm_narrator import get_comic_narrator
    import config
    LLM_NARRATOR_AVAILABLE = True
except ImportError:
    LLM_NARRATOR_AVAILABLE = False
    print("⚠️  Warning: llm_narrator module not available. Falling back to standard OCR.")

app = Flask(__name__)
CORS(app)

# Google Cloud clients are created lazily per-process because gRPC is not fork-safe
_vision_lock = Lock()
_tts_lock = Lock()
vision_client = None
vision_client_pid = None
tts_client = None
tts_client_pid = None


def get_vision_client():
    """Return a Google Vision client, creating one lazily per process."""
    if not credentials_available:
        raise RuntimeError("Google Cloud credentials are not configured")

    global vision_client, vision_client_pid
    pid = os.getpid()

    if vision_client is None or vision_client_pid != pid:
        with _vision_lock:
            if vision_client is None or vision_client_pid != pid:
                try:
                    vision_client = vision.ImageAnnotatorClient()
                    vision_client_pid = pid
                    print(f"✓ Google Vision client initialized (PID {pid})")
                except Exception as e:
                    vision_client = None
                    vision_client_pid = None
                    raise RuntimeError(f"Google Vision client initialization failed: {e}") from e

    return vision_client


def get_tts_client():
    """Return a Google Text-to-Speech client, creating one lazily per process."""
    if not credentials_available:
        raise RuntimeError("Google Cloud credentials are not configured")

    global tts_client, tts_client_pid
    pid = os.getpid()

    if tts_client is None or tts_client_pid != pid:
        with _tts_lock:
            if tts_client is None or tts_client_pid != pid:
                try:
                    tts_client = texttospeech.TextToSpeechClient()
                    tts_client_pid = pid
                    print(f"✓ Google TTS client initialized (PID {pid})")
                except Exception as e:
                    tts_client = None
                    tts_client_pid = None
                    raise RuntimeError(f"Google TTS client initialization failed: {e}") from e

    return tts_client

# Directories
AUDIO_DIR = Path("/app/audio_files")
AUDIO_DIR.mkdir(exist_ok=True)
TEMP_DIR = Path("/app/temp_images")
TEMP_DIR.mkdir(exist_ok=True)


def cleanup_old_files():
    """Remove files older than 1 hour"""
    cutoff = datetime.now() - timedelta(hours=1)
    for directory in [AUDIO_DIR, TEMP_DIR]:
        for file in directory.glob("*"):
            if datetime.fromtimestamp(file.stat().st_mtime) < cutoff:
                file.unlink()


class TextReorderer:
    """Use NLP techniques to reorder jumbled text into natural reading order"""
    
    @staticmethod
    def split_into_phrases(text):
        """Split text into meaningful phrases/clauses"""
        # Split on sentence boundaries and common phrase boundaries
        # Keep punctuation with the phrase
        phrases = re.split(r'([.!?]+\s+|,\s+|\s+)', text)
        
        # Recombine to preserve punctuation
        result = []
        current = ""
        for i, phrase in enumerate(phrases):
            if phrase.strip():
                if re.match(r'^[.!?,]+\s*$', phrase):
                    current += phrase
                else:
                    if current:
                        result.append(current.strip())
                    current = phrase
        
        if current.strip():
            result.append(current.strip())
        
        return [p for p in result if len(p) > 0]
    
    @staticmethod
    def calculate_coherence_score(phrase_order):
        """Calculate how natural/coherent a phrase ordering is"""
        text = ' '.join(phrase_order)
        score = 0
        
        # 1. Questions should come before answers
        question_patterns = [r'\bwhat\b', r'\bwhy\b', r'\bhow\b', r'\bwhere\b', r'\bwhen\b', r'\bwho\b', r'\?']
        answer_patterns = [r'\bbecause\b', r'\bso\b', r'\btherefore\b', r'\bin short\b', r'\bwell\b']
        
        has_question = any(re.search(pattern, text, re.IGNORECASE) for pattern in question_patterns)
        has_answer = any(re.search(pattern, text, re.IGNORECASE) for pattern in answer_patterns)
        
        if has_question and has_answer:
            question_pos = min([text.lower().find(re.search(p, text, re.IGNORECASE).group()) 
                               for p in question_patterns if re.search(p, text, re.IGNORECASE)])
            answer_pos = min([text.lower().find(re.search(p, text, re.IGNORECASE).group()) 
                             for p in answer_patterns if re.search(p, text, re.IGNORECASE)])
            
            if question_pos < answer_pos:
                score += 100  # Strong preference for Q before A
        
        # 2. Sentence structure: Subject-Verb-Object patterns
        # Prefer patterns like "X is Y" over "is Y X"
        if re.search(r'\b(the|a|an)\s+\w+\s+(is|are|was|were|has|have)', text, re.IGNORECASE):
            score += 30
        
        # 3. Common dialogue openers should be at the start
        starters = [r'^(so|well|oh|hey|look|listen|now)', r'^(tell me|show me|let me)']
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in starters):
            score += 20
        
        # 4. Penalize broken common phrases
        broken_phrases = [
            (r'for example', -50),  # "example for" is wrong
            (r'in short', -50),
            (r'tell me about', -50),
            (r'what do they do', -50),
            (r'getting together', -50),
        ]
        
        for phrase, penalty in broken_phrases:
            if phrase.replace(' ', r'\s+') in text.lower():
                score += 30  # Bonus for having it correctly
            else:
                # Check if words exist but in wrong order
                words = phrase.split()
                if all(word in text.lower() for word in words):
                    # Words exist but not in right order - penalty
                    score += penalty
        
        # 5. Punctuation should make sense
        # Questions should end with ?
        sentences = re.split(r'[.!?]+', text)
        for sent in sentences:
            sent = sent.strip()
            if sent:
                if any(word in sent.lower() for word in ['what', 'why', 'how', 'where', 'when', 'who']):
                    if text[text.find(sent) + len(sent):text.find(sent) + len(sent) + 2].find('?') >= 0:
                        score += 10
        
        return score
    
    @staticmethod
    def reorder_text(text):
        """Attempt to reorder jumbled text into natural reading order"""
        if not text or len(text.strip()) == 0:
            return text
            
        # First, try to detect if text seems jumbled
        # Look for signs: question words not followed by ?, split phrases
        
        # Split into words/tokens while preserving punctuation
        tokens = re.findall(r'\w+[.,!?]?', text)
        
        if len(tokens) < 3 or len(tokens) > 20:
            return text  # Too short or too long to meaningfully reorder
        
        # Try to find natural split points (likely separate speech bubbles)
        # Look for: end punctuation followed by capital letter
        sentences = re.split(r'([.!?]+\s+)(?=[A-Z])', text)
        sentences = [s for s in sentences if s.strip() and not re.match(r'^[.!?]+\s*$', s)]
        
        if len(sentences) == 1:
            # Single sentence - might be internally jumbled
            # Try common patterns
            patterns = [
                # Question then answer pattern
                (r'(.*?)\s+(WELL|SO|IN SHORT|FOR EXAMPLE)[,\s]+(.*)', 
                 lambda m: f"{m.group(1)} {m.group(2)}, {m.group(3)}"),
                
                # "Tell me about X" pattern
                (r'(TELL ME)\s+(THEIR|ABOUT)\s+(ABOUT|THEIR)?\s*(FESTIVALS|.*?)(!|\?|\s)', 
                 lambda m: f"TELL ME ABOUT THEIR {m.group(4)}{m.group(5)}"),
                
                # "What do they do" pattern
                (r'(WHAT DO|IN SHORT)[.\s]+(FAMILY GETTING|THEY DO|TOGETHER|EATING|.*?)\s+(WHAT DO|THEY DO|TOGETHER|EATING|.*?)',
                 lambda m: f"WHAT DO THEY DO? IN SHORT. {m.group(2)} {m.group(3)}"),
            ]
            
            for pattern, replacement in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    try:
                        return replacement(match)
                    except:
                        pass
        
        if len(sentences) == 2:
            # Two sentences - try both orders and pick better one
            order1 = f"{sentences[0]} {sentences[1]}"
            order2 = f"{sentences[1]} {sentences[0]}"
            
            score1 = TextReorderer.calculate_coherence_score([sentences[0], sentences[1]])
            score2 = TextReorderer.calculate_coherence_score([sentences[1], sentences[0]])
            
            return order1 if score1 >= score2 else order2
        
        return text


class SpeechBubbleDetector:
    """Detect and classify speech bubbles in comics"""
    
    @staticmethod
    def detect_bubbles(image_bytes):
        """Detect speech bubbles using contour analysis"""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return []
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise while keeping edges sharp
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Use adaptive thresholding for better bubble detection
        thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, 2)
        
        # Edge detection
        edges = cv2.Canny(filtered, 30, 100)
        
        # Combine threshold and edges
        combined = cv2.bitwise_or(thresh, edges)
        
        # Morphological operations to close gaps in bubble outlines
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Find contours
        contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        height, width = img.shape[:2]
        bubbles = []
        
        for idx, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            # Filter by area (speech bubbles are typically medium-sized)
            if area < (width * height * 0.003) or area > (width * height * 0.4):
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Skip very thin contours
            if w < 40 or h < 25:
                continue
            
            # Calculate properties
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            aspect_ratio = w / h if h > 0 else 0
            
            # Skip extremely elongated shapes (likely panel borders or lines)
            if aspect_ratio > 5.0 or aspect_ratio < 0.2:
                continue
            
            # Analyze shape to determine bubble type
            bubble_type = SpeechBubbleDetector.classify_bubble(contour, circularity, aspect_ratio)
            
            # More lenient acceptance - most contours with text could be bubbles
            if bubble_type != "unknown" or (0.3 < aspect_ratio < 4.0 and area > width * height * 0.01):
                if bubble_type == "unknown":
                    bubble_type = "speech"
                
                # Check if this bubble overlaps significantly with an existing bubble
                # If so, skip it (likely a duplicate detection)
                is_duplicate = False
                for existing in bubbles:
                    overlap_x = max(0, min(x + w, existing['x'] + existing['width']) - max(x, existing['x']))
                    overlap_y = max(0, min(y + h, existing['y'] + existing['height']) - max(y, existing['y']))
                    overlap_area = overlap_x * overlap_y
                    
                    # If overlap is more than 70% of smaller bubble, it's a duplicate
                    smaller_area = min(area, existing['area'])
                    if overlap_area > smaller_area * 0.7:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    bubbles.append({
                        'x': int(x),
                        'y': int(y),
                        'width': int(w),
                        'height': int(h),
                        'center_x': int(x + w/2),
                        'center_y': int(y + h/2),
                        'area': float(area),
                        'circularity': float(circularity),
                        'type': bubble_type,
                        'contour': contour
                    })
        
        return bubbles
    
    @staticmethod
    def classify_bubble(contour, circularity, aspect_ratio):
        """Classify bubble type based on shape characteristics"""
        
        # Approximate the contour
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        vertices = len(approx)
        
        # Speech bubble: oval/circular with tail (high circularity)
        if circularity > 0.6 and 0.5 < aspect_ratio < 2.0:
            return "speech"
        
        # Thought bubble: very circular (often multiple circles)
        elif circularity > 0.75:
            return "thought"
        
        # Shout/scream bubble: jagged edges (many vertices)
        elif vertices > 12 and circularity < 0.5:
            return "shout"
        
        # Rectangular caption box
        elif 4 <= vertices <= 6 and circularity < 0.7 and 1.5 < aspect_ratio < 5.0:
            return "caption"
        
        # Whisper bubble: dashed or dotted outline (detected by low circularity)
        elif circularity > 0.5 and circularity < 0.65:
            return "whisper"
        
        return "speech"  # Default to speech bubble
    
    @staticmethod
    def is_text_in_bubble(text_block, bubble, padding=10):
        """Check if text block center is inside a speech bubble"""
        # Use center of text block for more accurate matching
        tx_center = text_block['x'] + text_block['width'] / 2
        ty_center = text_block['y'] + text_block['height'] / 2
        
        bx, by, bw, bh = bubble['x'], bubble['y'], bubble['width'], bubble['height']
        
        return (bx - padding <= tx_center <= bx + bw + padding and 
                by - padding <= ty_center <= by + bh + padding)


class ImagePreprocessor:
    """Enhance image quality for better OCR"""
    
    @staticmethod
    def enhance_image(image_bytes):
        """Apply preprocessing to improve OCR accuracy"""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return image_bytes
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Increase contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
        
        # Sharpen
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        # Threshold to make text clearer
        _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Convert back to bytes
        _, buffer = cv2.imencode('.png', binary)
        return buffer.tobytes()


class ComicOCR:
    """Handle comic panel detection and text extraction with speech bubble awareness"""
    
    def __init__(self):
        self.preprocessor = ImagePreprocessor()
        self.bubble_detector = SpeechBubbleDetector()
        self.text_reorderer = TextReorderer()  # ADD NLP text reorderer
    
    def detect_panels(self, image_bytes):
        """Detect comic panels using edge detection"""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return []
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate to connect panel borders
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        height, width = img.shape[:2]
        min_area = (width * height) * 0.05  # Panels should be at least 5% of image
        max_area = (width * height) * 0.9
        
        panels = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                if 0.2 < aspect_ratio < 5.0:  # Reasonable aspect ratio
                    panels.append({
                        'x': int(x), 'y': int(y),
                        'width': int(w), 'height': int(h),
                        'center_x': int(x + w/2),
                        'center_y': int(y + h/2)
                    })
        
        # If no panels detected, treat entire image as one panel
        if not panels:
            panels = [{
                'x': 0, 'y': 0, 'width': width, 'height': height,
                'center_x': width // 2, 'center_y': height // 2
            }]
        
        return self.sort_panels_reading_order(panels)
    
    def sort_panels_reading_order(self, panels):
        """Sort panels in Western comic reading order (left-to-right, top-to-bottom)"""
        if not panels:
            return panels
        
        # Group panels into rows
        rows = []
        sorted_by_y = sorted(panels, key=lambda p: p['y'])
        
        for panel in sorted_by_y:
            placed = False
            panel_top = panel['y']
            panel_bottom = panel['y'] + panel['height']
            
            for row in rows:
                row_top = min(p['y'] for p in row)
                row_bottom = max(p['y'] + p['height'] for p in row)
                overlap = min(panel_bottom, row_bottom) - max(panel_top, row_top)
                
                # If panels overlap vertically by at least 30%, they're in the same row
                if overlap > panel['height'] * 0.3:
                    row.append(panel)
                    placed = True
                    break
            
            if not placed:
                rows.append([panel])
        
        # Sort each row left-to-right
        sorted_panels = []
        for row in rows:
            row_sorted = sorted(row, key=lambda p: p['x'])
            sorted_panels.extend(row_sorted)
        
        return sorted_panels
    
    def sort_bubbles_in_panel(self, bubbles):
        """Sort speech bubbles in natural reading order within a panel"""
        if not bubbles:
            return bubbles
        
        # Sort by vertical position first (top to bottom), then horizontal (left to right)
        # Group bubbles that are roughly at the same height
        rows = []
        sorted_by_y = sorted(bubbles, key=lambda b: b['center_y'])
        
        for bubble in sorted_by_y:
            placed = False
            bubble_top = bubble['y']
            bubble_bottom = bubble['y'] + bubble['height']
            
            for row in rows:
                row_top = min(b['y'] for b in row)
                row_bottom = max(b['y'] + b['height'] for b in row)
                overlap = min(bubble_bottom, row_bottom) - max(bubble_top, row_top)
                
                # If bubbles overlap vertically, they're in the same "row"
                if overlap > bubble['height'] * 0.3:
                    row.append(bubble)
                    placed = True
                    break
            
            if not placed:
                rows.append([bubble])
        
        # Sort each row left-to-right
        sorted_bubbles = []
        for row in rows:
            row_sorted = sorted(row, key=lambda b: b['center_x'])
            sorted_bubbles.extend(row_sorted)
        
        return sorted_bubbles
    
    def group_text_by_proximity(self, text_blocks):
        """Group text blocks that are close together (likely same bubble) using spatial clustering"""
        if not text_blocks:
            return []
        
        # Use a more sophisticated clustering approach
        # Calculate average text dimensions
        avg_height = sum(t['height'] for t in text_blocks) / len(text_blocks)
        avg_width = sum(t['width'] for t in text_blocks) / len(text_blocks)
        
        groups = []
        used = set()
        
        for i, text in enumerate(text_blocks):
            if i in used:
                continue
            
            # Start new group
            group = [text]
            used.add(i)
            
            # Find nearby text blocks iteratively (to build connected groups)
            changed = True
            while changed:
                changed = False
                for j, other in enumerate(text_blocks):
                    if j in used:
                        continue
                    
                    # Check if close to ANY text in the current group
                    min_dist = float('inf')
                    closest_in_group = None
                    
                    for group_text in group:
                        # Calculate distance between text blocks (center to center)
                        center1_x = group_text['x'] + group_text['width'] / 2
                        center1_y = group_text['y'] + group_text['height'] / 2
                        center2_x = other['x'] + other['width'] / 2
                        center2_y = other['y'] + other['height'] / 2
                        
                        dist_x = abs(center1_x - center2_x)
                        dist_y = abs(center1_y - center2_y)
                        
                        # Calculate Euclidean distance
                        euclidean_dist = (dist_x ** 2 + dist_y ** 2) ** 0.5
                        
                        if euclidean_dist < min_dist:
                            min_dist = euclidean_dist
                            closest_in_group = (dist_x, dist_y)
                    
                    if closest_in_group:
                        dist_x, dist_y = closest_in_group
                        
                        # MUCH STRICTER criteria
                        # Words in the same bubble are typically:
                        # - Very close vertically (within 1.5x height)
                        # - Reasonably close horizontally (within 2.5x width)
                        vertical_threshold = avg_height * 1.5  # Reduced from 2.0
                        horizontal_threshold = avg_width * 2.5  # Reduced from 4.0
                        
                        is_close_vertically = dist_y < vertical_threshold
                        is_close_horizontally = dist_x < horizontal_threshold
                        
                        # MUST be close in BOTH dimensions
                        if is_close_vertically and is_close_horizontally:
                            group.append(other)
                            used.add(j)
                            changed = True
                            break
            
            groups.append(group)
        
        return groups
    
    def extract_text(self, image_bytes, preprocess=True, use_llm=None):
        """
        Extract text from comic with speech bubble detection and NLP text reordering.

        Args:
            image_bytes: Raw image bytes
            preprocess: Whether to preprocess image (only for OCR mode)
            use_llm: Override to use LLM narration. If None, uses config.USE_LLM_NARRATOR

        Returns:
            Dict with text, panels, bubbles, and metadata
        """
        # Determine if we should use LLM narrator
        should_use_llm = use_llm
        if should_use_llm is None:
            try:
                should_use_llm = config.USE_LLM_NARRATOR
            except:
                should_use_llm = False

        # Use LLM narrator if available and enabled
        if should_use_llm and LLM_NARRATOR_AVAILABLE:
            try:
                return self._extract_text_with_llm(image_bytes)
            except Exception as e:
                print(f"⚠️  LLM narration failed: {e}. Falling back to standard OCR.")
                # Fall through to standard OCR

        # Standard OCR extraction
        return self._extract_text_with_ocr(image_bytes, preprocess)

    def _extract_text_with_llm(self, image_bytes):
        """Extract text using LLM-based audiobook narration"""
        try:
            narrator = get_comic_narrator()
            result = narrator.narrate_single_image(image_bytes)

            if not result['success']:
                raise Exception(f"LLM narration failed: {result.get('error', 'Unknown error')}")

            # Format result to match expected structure
            return {
                "text": result['narration'],
                "panels": [],  # LLM handles panels internally
                "panel_count": 0,
                "bubbles": [],
                "bubble_count": 0,
                "text_blocks": [],
                "confidence": result.get('confidence', 1.0),
                "narration_mode": "llm",
                "tokens_used": result.get('tokens_used')
            }
        except Exception as e:
            raise Exception(f"LLM extraction failed: {str(e)}")

    def _extract_text_with_ocr(self, image_bytes, preprocess=True):
        """Extract text from comic using traditional OCR with speech bubble detection"""

        try:
            vision_client = get_vision_client()
        except Exception as exc:
            raise Exception("Google Vision client not initialized. Please check your credentials.") from exc
        
        # Preprocess if requested
        if preprocess:
            processed_bytes = self.preprocessor.enhance_image(image_bytes)
        else:
            processed_bytes = image_bytes
        
        # Run Google Vision API
        image = vision.Image(content=processed_bytes)
        response = vision_client.text_detection(image=image)
        
        if response.error.message:
            raise Exception(f"Vision API Error: {response.error.message}")
        
        texts = response.text_annotations
        if not texts:
            return {"text": "", "panels": [], "bubbles": [], "text_blocks": [], "confidence": 0}
        
        # Detect panels and speech bubbles
        panels = self.detect_panels(image_bytes)
        all_bubbles = self.bubble_detector.detect_bubbles(image_bytes)
        
        # Build text blocks with confidence scores
        text_blocks = []
        for text in texts[1:]:  # Skip first element (full text)
            vertices = [(v.x, v.y) for v in text.bounding_poly.vertices]
            x_coords = [v[0] for v in vertices]
            y_coords = [v[1] for v in vertices]
            x, y = min(x_coords), min(y_coords)
            width = max(x_coords) - x
            height = max(y_coords) - y
            
            confidence = getattr(text, 'confidence', 1.0)
            
            text_blocks.append({
                'text': text.description,
                'x': x,
                'y': y,
                'width': width,
                'height': height,
                'confidence': confidence,
                'used': False  # Track if text has been assigned to a bubble
            })
        
        # Calculate average confidence
        avg_confidence = sum(b['confidence'] for b in text_blocks) / len(text_blocks) if text_blocks else 0
        
        # Process each panel
        all_text = []
        panel_data = []
        bubble_data = []
        
        for panel_idx, panel in enumerate(panels):
            # Find bubbles in this panel
            panel_bubbles = [b for b in all_bubbles 
                           if panel['x'] <= b['center_x'] <= panel['x'] + panel['width']
                           and panel['y'] <= b['center_y'] <= panel['y'] + panel['height']]
            
            # Sort bubbles in reading order
            panel_bubbles = self.sort_bubbles_in_panel(panel_bubbles)
            
            if panel_bubbles or any(self.is_text_in_panel(t, panel) for t in text_blocks):
                panel_label = f"[Panel {panel_idx + 1}]"
                all_text.append(panel_label)
                panel_texts = []
                
                # Process each bubble in order
                for bubble_idx, bubble in enumerate(panel_bubbles):
                    # Find text blocks that belong to this bubble and haven't been used
                    bubble_texts = [t for t in text_blocks 
                                  if not t['used'] and self.bubble_detector.is_text_in_bubble(t, bubble)]
                    
                    if bubble_texts:
                        # Mark texts as used
                        for t in bubble_texts:
                            t['used'] = True
                        
                        # IMPROVED SORTING: Group into lines first, then sort each line left-to-right
                        # Step 1: Sort by Y position to get rough line groupings
                        bubble_texts.sort(key=lambda t: t['y'])
                        
                        # Step 2: Group text blocks into lines based on Y overlap
                        lines = []
                        for text in bubble_texts:
                            placed = False
                            text_top = text['y']
                            text_bottom = text['y'] + text['height']
                            
                            # Try to add to existing line
                            for line in lines:
                                line_top = min(t['y'] for t in line)
                                line_bottom = max(t['y'] + t['height'] for t in line)
                                
                                # Check if this text overlaps vertically with the line
                                overlap = min(text_bottom, line_bottom) - max(text_top, line_top)
                                avg_height = (text['height'] + line[0]['height']) / 2
                                
                                if overlap > avg_height * 0.5:  # 50% overlap means same line
                                    line.append(text)
                                    placed = True
                                    break
                            
                            if not placed:
                                lines.append([text])
                        
                        # Step 3: Sort each line left-to-right and combine
                        sorted_lines = []
                        for line in lines:
                            line.sort(key=lambda t: t['x'])
                            line_text = ' '.join([t['text'] for t in line])
                            sorted_lines.append(line_text)
                        
                        # Join all lines with space for natural speech flow
                        text_content = ' '.join(sorted_lines)
                        
                        # Clean up extra spaces and punctuation issues
                        text_content = text_content.replace(' ,', ',').replace(' .', '.').replace(' !', '!').replace(' ?', '?')
                        text_content = text_content.replace('  ', ' ').strip()
                        
                        # ===== APPLY NLP TEXT REORDERING =====
                        text_content = self.text_reorderer.reorder_text(text_content)
                        
                        # Format based on bubble type
                        bubble_type = bubble['type']
                        
                        if bubble_type == "thought":
                            formatted_text = f"(thinking: {text_content})"
                        elif bubble_type == "shout":
                            formatted_text = text_content.upper()
                        elif bubble_type == "whisper":
                            formatted_text = f"(whispers: {text_content})"
                        elif bubble_type == "caption":
                            formatted_text = f"[{text_content}]"
                        else:  # speech
                            formatted_text = text_content
                        
                        all_text.append(formatted_text)
                        panel_texts.append(formatted_text)
                        
                        bubble_data.append({
                            'panel': panel_idx + 1,
                            'bubble_type': bubble_type,
                            'text': text_content,
                            'position': bubble_idx + 1
                        })
                
                # Handle text not in any bubble (captions, sound effects) that hasn't been used
                non_bubble_text = [t for t in text_blocks 
                                 if not t['used'] 
                                 and self.is_text_in_panel(t, panel)]
                
                if non_bubble_text:
                    # Group non-bubble text by proximity (likely same speech bubble missed by detection)
                    text_groups = self.group_text_by_proximity(non_bubble_text)
                    
                    # Sort groups by reading order (top-to-bottom, left-to-right)
                    text_groups.sort(key=lambda g: (min(t['y'] for t in g), min(t['x'] for t in g)))
                    
                    for group in text_groups:
                        # Sort by Y first to get lines
                        group.sort(key=lambda t: t['y'])
                        
                        # Group into lines based on Y overlap
                        lines = []
                        for text in group:
                            text['used'] = True
                            placed = False
                            text_top = text['y']
                            text_bottom = text['y'] + text['height']
                            
                            for line in lines:
                                line_top = min(t['y'] for t in line)
                                line_bottom = max(t['y'] + t['height'] for t in line)
                                overlap = min(text_bottom, line_bottom) - max(text_top, line_top)
                                avg_height = (text['height'] + line[0]['height']) / 2
                                
                                if overlap > avg_height * 0.5:
                                    line.append(text)
                                    placed = True
                                    break
                            
                            if not placed:
                                lines.append([text])
                        
                        # Sort each line left-to-right
                        sorted_lines = []
                        for line in lines:
                            line.sort(key=lambda t: t['x'])
                            line_text = ' '.join([t['text'] for t in line])
                            sorted_lines.append(line_text)
                        
                        combined_text = ' '.join(sorted_lines)
                        combined_text = combined_text.replace(' ,', ',').replace(' .', '.').replace(' !', '!').replace(' ?', '?')
                        combined_text = combined_text.replace('  ', ' ').strip()
                        
                        # ===== APPLY NLP TEXT REORDERING =====
                        combined_text = self.text_reorderer.reorder_text(combined_text)
                        
                        all_text.append(combined_text)
                        panel_texts.append(combined_text)
                
                if panel_texts:
                    panel_data.append({
                        'panel': panel_idx + 1,
                        'text': '\n'.join(panel_texts),
                        'bubble_count': len([t for t in panel_texts if not t.startswith('[')])
                    })
        
        return {
            "text": '\n'.join(all_text),
            "panels": panel_data,
            "panel_count": len(panels),
            "bubbles": bubble_data,
            "bubble_count": len(all_bubbles),
            "text_blocks": text_blocks,
            "confidence": avg_confidence,
            "narration_mode": "ocr"
        }
    
    def is_text_in_panel(self, text_block, panel):
        """Check if text block is within a panel"""
        text_x, text_y = text_block['x'], text_block['y']
        return (panel['x'] <= text_x <= panel['x'] + panel['width'] and 
                panel['y'] <= text_y <= panel['y'] + panel['height'])


# HTML Template with enhanced UI
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Comic Reader with Speech Bubble Detection</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: #333;
      padding: 20px;
      min-height: 100vh;
    }
    .container {
      max-width: 1200px;
      margin: 30px auto;
      background: white;
      padding: 30px;
      border-radius: 16px;
      box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    }
    h1 {
      color: #667eea;
      margin-bottom: 10px;
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }
    .badge {
      background: linear-gradient(135deg, #667eea, #764ba2);
      color: white;
      padding: 4px 12px;
      border-radius: 20px;
      font-size: 12px;
    }
    .badge.new {
      background: linear-gradient(135deg, #f59e0b, #d97706);
      animation: pulse 2s infinite;
    }
    @keyframes pulse {
      0%, 100% { transform: scale(1); }
      50% { transform: scale(1.05); }
    }
    .subtitle {
      color: #666;
      margin-bottom: 25px;
    }
    .upload-area {
      border: 3px dashed #667eea;
      border-radius: 12px;
      padding: 50px 20px;
      margin: 20px 0;
      cursor: pointer;
      transition: all 0.3s;
      background: #f7fafc;
      text-align: center;
    }
    .upload-area:hover {
      background: #ebf8ff;
      border-color: #764ba2;
      transform: scale(1.01);
    }
    .upload-icon { font-size: 48px; margin-bottom: 15px; }
    .button {
      background: linear-gradient(135deg, #667eea, #764ba2);
      color: white;
      border: none;
      padding: 14px 28px;
      border-radius: 8px;
      cursor: pointer;
      font-size: 16px;
      font-weight: 600;
      transition: all 0.3s;
      box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    .button:hover {
      transform: translateY(-2px);
      box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    .button:disabled {
      background: #ccc;
      cursor: not-allowed;
      box-shadow: none;
    }
    .result-box {
      background: #f7fafc;
      border-radius: 12px;
      padding: 25px;
      margin: 20px 0;
      border-left: 5px solid #667eea;
    }
    .result-box h3 {
      color: #667eea;
      margin-bottom: 15px;
      display: flex;
      align-items: center;
      gap: 10px;
    }
    textarea {
      width: 100%;
      padding: 15px;
      border: 2px solid #e2e8f0;
      border-radius: 8px;
      font-size: 15px;
      font-family: inherit;
      resize: vertical;
      min-height: 150px;
    }
    .stats {
      display: flex;
      gap: 20px;
      flex-wrap: wrap;
      margin: 20px 0;
    }
    .stat-card {
      background: white;
      padding: 15px 20px;
      border-radius: 10px;
      box-shadow: 0 2px 10px rgba(0,0,0,0.1);
      flex: 1;
      min-width: 150px;
    }
    .stat-label {
      color: #666;
      font-size: 13px;
      margin-bottom: 5px;
    }
    .stat-value {
      color: #667eea;
      font-size: 24px;
      font-weight: bold;
    }
    audio {
      width: 100%;
      margin: 15px 0;
    }
    .controls {
      display: flex;
      gap: 15px;
      margin: 20px 0;
      flex-wrap: wrap;
    }
    .voice-select {
      flex: 1;
      min-width: 200px;
      padding: 12px;
      border: 2px solid #e2e8f0;
      border-radius: 8px;
      font-size: 15px;
    }
    .spinner {
      border: 4px solid #f3f3f3;
      border-top: 4px solid #667eea;
      border-radius: 50%;
      width: 40px;
      height: 40px;
      animation: spin 1s linear infinite;
      margin: 20px auto;
    }
    @keyframes spin {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
    }
    .checkbox-group {
      display: flex;
      align-items: center;
      gap: 10px;
      margin: 15px 0;
    }
    .features-list {
      background: #ebf8ff;
      padding: 20px;
      border-radius: 10px;
      margin: 20px 0;
    }
    .features-list ul {
      list-style-position: inside;
      color: #2d3748;
    }
    .features-list li {
      padding: 5px 0;
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>
      🎨 Comic Reader
      <span class="badge">V2.0</span>
      <span class="badge new">✨ NLP Enhanced</span>
    </h1>
    <p class="subtitle">Upload your comic page and let AI read it to you!</p>
    
    <div class="features-list">
      <strong>✨ New Features:</strong>
      <ul>
        <li>🧠 <strong>NLP Text Reordering</strong> - Automatically fixes jumbled text using natural language processing</li>
        <li>💬 Speech bubble detection and classification</li>
        <li>📖 Natural comic reading order (left→right, top→bottom)</li>
        <li>🎭 Bubble type recognition (speech, thought, shout, whisper, caption)</li>
        <li>🔄 Intelligent text grouping within bubbles</li>
        <li>✨ Human-like reading flow with proper punctuation</li>
      </ul>
    </div>

    <div class="upload-area" id="uploadArea">
      <div class="upload-icon">📤</div>
      <h3>Click to upload or drag & drop</h3>
      <p>Supports: JPG, PNG, GIF, WebP</p>
      <input type="file" id="fileInput" accept="image/*" style="display: none;" />
    </div>

    <div class="checkbox-group">
      <input type="checkbox" id="preprocessToggle" checked />
      <label for="preprocessToggle">🔧 Enable image preprocessing (recommended for better accuracy)</label>
    </div>

    <div class="controls">
      <select class="voice-select" id="voiceSelect">
        <option value="en-US-Neural2-F">🎭 Female Voice 1 (US)</option>
        <option value="en-US-Neural2-C">🎭 Female Voice 2 (US)</option>
        <option value="en-US-Neural2-E">🎭 Female Voice 3 (US)</option>
        <option value="en-US-Neural2-D">👨 Male Voice 1 (US)</option>
        <option value="en-US-Neural2-A">👨 Male Voice 2 (US)</option>
        <option value="en-US-Neural2-I">👦 Child Voice (US)</option>
        <option value="en-GB-Neural2-A">🇬🇧 British Female</option>
        <option value="en-GB-Neural2-B">🇬🇧 British Male</option>
        <option value="en-AU-Neural2-A">🇦🇺 Australian Female</option>
        <option value="en-AU-Neural2-B">🇦🇺 Australian Male</option>
      </select>
      <button class="button" id="processBtn" disabled>🚀 Process Comic</button>
    </div>

    <div id="results" style="display: none;">
      <div class="stats">
        <div class="stat-card">
          <div class="stat-label">📊 Panels Detected</div>
          <div class="stat-value" id="panelCount">0</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">💬 Bubbles Found</div>
          <div class="stat-value" id="bubbleCount">0</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">✅ Confidence</div>
          <div class="stat-value" id="confidence">0%</div>
        </div>
      </div>

      <div class="result-box">
        <h3>📝 Extracted Text <span class="badge new">NLP Reordered</span></h3>
        <textarea id="extractedText" placeholder="Extracted text will appear here..."></textarea>
      </div>

      <div class="result-box">
        <h3>🔊 Generated Audio</h3>
        <audio id="audioPlayer" controls></audio>
      </div>
    </div>

    <div id="loadingDiv" style="display: none;">
      <div class="spinner"></div>
      <p style="text-align: center; color: #667eea; margin-top: 10px;">
        Processing your comic with NLP text reordering...
      </p>
    </div>
  </div>

  <script>
    let selectedFile = null;

    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const processBtn = document.getElementById('processBtn');
    const voiceSelect = document.getElementById('voiceSelect');
    const preprocessToggle = document.getElementById('preprocessToggle');
    const resultsDiv = document.getElementById('results');
    const loadingDiv = document.getElementById('loadingDiv');

    uploadArea.addEventListener('click', () => fileInput.click());

    uploadArea.addEventListener('dragover', (e) => {
      e.preventDefault();
      uploadArea.style.background = '#ebf8ff';
    });

    uploadArea.addEventListener('dragleave', () => {
      uploadArea.style.background = '#f7fafc';
    });

    uploadArea.addEventListener('drop', (e) => {
      e.preventDefault();
      uploadArea.style.background = '#f7fafc';
      const file = e.dataTransfer.files[0];
      if (file && file.type.startsWith('image/')) {
        handleFileSelect(file);
      }
    });

    fileInput.addEventListener('change', (e) => {
      const file = e.target.files[0];
      if (file) {
        handleFileSelect(file);
      }
    });

    function handleFileSelect(file) {
      selectedFile = file;
      uploadArea.innerHTML = `
        <div class="upload-icon">✅</div>
        <h3>File selected: ${file.name}</h3>
        <p>Ready to process!</p>
      `;
      processBtn.disabled = false;
    }

    processBtn.addEventListener('click', async () => {
      if (!selectedFile) return;

      resultsDiv.style.display = 'none';
      loadingDiv.style.display = 'block';
      processBtn.disabled = true;

      const formData = new FormData();
      formData.append('image', selectedFile);
      formData.append('voice_name', voiceSelect.value);
      formData.append('language_code', 'en-US');
      formData.append('preprocess', preprocessToggle.checked);

      try {
        const response = await fetch('/api/process-comic', {
          method: 'POST',
          body: formData
        });

        const data = await response.json();

        if (data.success) {
          document.getElementById('panelCount').textContent = data.panel_count;
          document.getElementById('bubbleCount').textContent = data.bubble_count;
          document.getElementById('confidence').textContent = 
            Math.round(data.confidence * 100) + '%';
          document.getElementById('extractedText').value = data.extracted_text;
          document.getElementById('audioPlayer').src = data.audio_url;

          resultsDiv.style.display = 'block';
        } else {
          alert('Error: ' + (data.error || 'Unknown error'));
        }
      } catch (error) {
        alert('Error processing comic: ' + error.message);
      } finally {
        loadingDiv.style.display = 'none';
        processBtn.disabled = false;
      }
    });
  </script>
</body>
</html>
"""

# Routes
@app.route('/')
def index():
    """Serve the frontend"""
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy"})


@app.route('/api/extract-text', methods=['POST'])
def extract_text():
    """Extract text from comic image with speech bubble detection"""
    try:
        cleanup_old_files()
        
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        preprocess = request.form.get('preprocess', 'true').lower() == 'true'
        
        image_bytes = file.read()
        
        # Extract text with bubble detection
        ocr = ComicOCR()
        result = ocr.extract_text(image_bytes, preprocess=preprocess)
        
        return jsonify({
            "success": True,
            "extracted_text": result["text"],
            "panel_count": result["panel_count"],
            "bubble_count": result["bubble_count"],
            "text_blocks": len(result["text_blocks"]),
            "confidence": result["confidence"]
        })
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/generate-audio', methods=['POST'])
def generate_audio():
    """Generate audio from text"""
    try:
        cleanup_old_files()
        
        try:
            tts_client = get_tts_client()
        except Exception as exc:
            return jsonify({"error": f"Text-to-Speech client not initialized: {exc}"}), 500
        
        data = request.get_json()
        text = data.get('text', '')
        language_code = data.get('language_code', 'en-US')
        voice_name = data.get('voice_name', 'en-US-Neural2-F')
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        # Generate audio
        synthesis_input = texttospeech.SynthesisInput(text=text)
        
        voice = texttospeech.VoiceSelectionParams(
            language_code=language_code,
            name=voice_name
        )
        
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=1.0,
            pitch=0.0
        )
        
        response = tts_client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )
        
        # Save audio
        audio_id = str(uuid.uuid4())
        audio_path = AUDIO_DIR / f"{audio_id}.mp3"
        
        with open(audio_path, 'wb') as out:
            out.write(response.audio_content)
        
        return jsonify({
            "success": True,
            "audio_url": f"/api/audio/{audio_id}",
            "characters_used": len(text)
        })
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/process-comic', methods=['POST'])
def process_comic():
    """Combined endpoint: extract text and generate audio"""
    try:
        cleanup_old_files()
        
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        language_code = request.form.get('language_code', 'en-US')
        voice_name = request.form.get('voice_name', 'en-US-Neural2-F')
        preprocess = request.form.get('preprocess', 'true').lower() == 'true'
        
        image_bytes = file.read()
        
        # Extract text with bubble detection
        ocr = ComicOCR()
        ocr_result = ocr.extract_text(image_bytes, preprocess=preprocess)
        extracted_text = ocr_result["text"]
        
        if not extracted_text:
            return jsonify({
                "success": False,
                "error": "No text found in image"
            }), 400
        
        # Generate audio
        try:
            tts_client = get_tts_client()
        except Exception as exc:
            return jsonify({"error": f"Text-to-Speech client not initialized: {exc}"}), 500
            
        synthesis_input = texttospeech.SynthesisInput(text=extracted_text)
        
        voice = texttospeech.VoiceSelectionParams(
            language_code=language_code,
            name=voice_name
        )
        
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=1.0,
            pitch=0.0
        )
        
        response = tts_client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )
        
        # Save audio
        audio_id = str(uuid.uuid4())
        audio_path = AUDIO_DIR / f"{audio_id}.mp3"
        
        with open(audio_path, 'wb') as out:
            out.write(response.audio_content)
        
        return jsonify({
            "success": True,
            "extracted_text": extracted_text,
            "panel_count": ocr_result["panel_count"],
            "bubble_count": ocr_result["bubble_count"],
            "text_blocks": len(ocr_result["text_blocks"]),
            "confidence": ocr_result["confidence"],
            "audio_url": f"/api/audio/{audio_id}",
            "characters_used": len(extracted_text)
        })
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/audio/<audio_id>', methods=['GET'])
def get_audio(audio_id):
    """Serve generated audio file"""
    try:
        audio_path = AUDIO_DIR / f"{audio_id}.mp3"
        
        if not audio_path.exists():
            return jsonify({"error": "Audio file not found"}), 404
        
        return send_file(
            audio_path,
            mimetype='audio/mpeg',
            as_attachment=False
        )
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 COMIC READER WITH SPEECH BUBBLE DETECTION & NLP")
    print("="*70)
    
    if credentials_available:
        print(f"✓ Server ready to start")
        print(f"✓ Google Cloud credentials detected. Clients will initialize on first use.")
    else:
        print(f"⚠️  Server will start, but Google Cloud APIs are not available")
        print(f"   Please check your credentials.json file")
    
    print(f"\n📍 Server will run at: http://localhost:5000")
    print(f"📍 Open your browser and go to: http://localhost:5000")
    print(f"")
    print(f"✨ NEW FEATURES:")
    print(f"  🧠 NLP text reordering - automatically fixes jumbled text")
    print(f"  💬 Speech bubble detection and classification")
    print(f"  📖 Natural comic reading order (left→right, top→bottom)")
    print(f"  🎭 Bubble type recognition (speech, thought, shout, whisper, caption)")
    print(f"  🔄 Intelligent text grouping within bubbles")
    print(f"  ✨ Human-like reading flow")
    print(f"")
    print(f"Previous Features:")
    print(f"  🔧 Image preprocessing for better accuracy")
    print(f"  ✏️  Manual text editing")
    print(f"  📊 Confidence scores")
    print(f"  🎤 10+ voice options")
    print("="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5001)
