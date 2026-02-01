"""
Image Analysis Module
Provides OCR (text extraction from images) and CLIP-based image understanding
"""
import logging
import io
import base64
from typing import Dict, Any, Optional, List
from pathlib import Path

import torch
import requests
from PIL import Image

logger = logging.getLogger(__name__)

# Global models
_ocr_reader = None
_clip_model = None
_clip_processor = None
_device = None


def get_device() -> str:
    """Get the best available device"""
    global _device
    if _device is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
    return _device


def load_ocr_model():
    """Load EasyOCR model for text extraction - optimized for speed"""
    global _ocr_reader
    
    if _ocr_reader is not None:
        return _ocr_reader
    
    try:
        import easyocr
        logger.info("Loading OCR model (GPU-accelerated)...")
        # Use GPU, disable model storage warning
        _ocr_reader = easyocr.Reader(
            ['en'], 
            gpu=torch.cuda.is_available(),
            model_storage_directory=None,  # Use default
            download_enabled=True
        )
        logger.info(f"  ✅ OCR model loaded on {'GPU' if torch.cuda.is_available() else 'CPU'}")
        return _ocr_reader
    except Exception as e:
        logger.error(f"Failed to load OCR model: {e}")
        return None


def load_clip_model():
    """Load CLIP model for image understanding"""
    global _clip_model, _clip_processor
    
    if _clip_model is not None:
        return _clip_model, _clip_processor
    
    try:
        from transformers import CLIPProcessor, CLIPModel
        
        logger.info("Loading CLIP model...")
        device = get_device()
        
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _clip_model.to(device)
        _clip_model.eval()
        
        logger.info(f"  ✅ CLIP model loaded on {device}")
        return _clip_model, _clip_processor
    except Exception as e:
        logger.error(f"Failed to load CLIP model: {e}")
        return None, None


def fetch_image(image_source: str) -> Optional[Image.Image]:
    """
    Fetch image from URL or decode from base64
    
    Args:
        image_source: URL string or base64 encoded image data
    
    Returns:
        PIL Image or None if failed
    """
    try:
        if image_source.startswith('data:image'):
            # Base64 encoded image
            header, data = image_source.split(',', 1)
            image_data = base64.b64decode(data)
            return Image.open(io.BytesIO(image_data)).convert('RGB')
        
        elif image_source.startswith('http'):
            # URL
            response = requests.get(image_source, timeout=10, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content)).convert('RGB')
        
        elif Path(image_source).exists():
            # Local file path
            return Image.open(image_source).convert('RGB')
        
        else:
            logger.error(f"Unknown image source format: {image_source[:50]}...")
            return None
            
    except Exception as e:
        logger.error(f"Failed to fetch image: {e}")
        return None


def preprocess_image_for_ocr(image: Image.Image) -> Image.Image:
    """
    Fast image preprocessing for OCR
    - Only resize if necessary
    - Minimal processing for speed
    """
    # Get image dimensions
    width, height = image.size
    
    # Only upscale very small images
    min_dimension = min(width, height)
    if min_dimension < 600:
        scale = 600 / min_dimension
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = image.resize((new_width, new_height), Image.Resampling.BILINEAR)  # Faster than LANCZOS
    
    # Downscale very large images (faster processing)
    max_dimension = max(width, height)
    if max_dimension > 2000:
        scale = 2000 / max_dimension
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = image.resize((new_width, new_height), Image.Resampling.BILINEAR)
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return image


def clean_ocr_text(text: str) -> str:
    """
    Clean up OCR output to fix common errors
    """
    import re
    
    if not text:
        return text
    
    # Remove excessive special characters that are likely OCR errors
    # Keep alphanumeric, basic punctuation, and common symbols
    text = re.sub(r'[^\w\s.,!?\'\"@#$%&*()[\]{}<>:;/\\|+=\-_~`]', '', text)
    
    # Fix common OCR misreads
    replacements = {
        ' l ': ' I ',      # lowercase L to I
        ' 0 ': ' O ',      # zero to O in words
        '|': 'I',          # pipe to I
        ' rn ': ' m ',     # rn often misread as m
        '  ': ' ',         # double spaces
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Remove very short fragments that are likely noise (1-2 chars unless they're meaningful)
    words = text.split()
    cleaned_words = []
    for word in words:
        # Keep word if it's meaningful
        if len(word) > 2 or word.lower() in ['a', 'i', 'to', 'is', 'it', 'in', 'on', 'or', 'an', 'as', 'at', 'be', 'by', 'do', 'go', 'he', 'if', 'me', 'my', 'no', 'of', 'ok', 'so', 'up', 'us', 'we']:
            cleaned_words.append(word)
    
    return ' '.join(cleaned_words).strip()


def extract_text_from_image(image: Image.Image) -> Dict[str, Any]:
    """
    Extract text from image using OCR with preprocessing for better accuracy
    
    Returns:
        {
            'text': str,           # Combined extracted text
            'segments': List[Dict], # Individual text segments with confidence
            'confidence': float     # Average confidence
        }
    """
    reader = load_ocr_model()
    
    if reader is None:
        return {'text': '', 'segments': [], 'confidence': 0.0}
    
    try:
        import numpy as np
        
        # Preprocess image for better OCR
        processed_image = preprocess_image_for_ocr(image)
        image_np = np.array(processed_image)
        
        # Perform OCR with SPEED-OPTIMIZED parameters
        results = reader.readtext(
            image_np,
            paragraph=True,           # Group text for faster processing
            min_size=20,              # Skip tiny text (faster)
            contrast_ths=0.1,
            adjust_contrast=0.5,
            text_threshold=0.6,       # Slightly lower threshold
            low_text=0.3,
            link_threshold=0.3,
            decoder='greedy',         # Fast greedy decoder
            beamWidth=3,              # Smaller beam = faster
            batch_size=4,             # Batch processing
            workers=0                 # No multiprocessing (Windows compatible)
        )
        
        # Parse results: [(bbox, text, confidence), ...]
        segments = []
        texts = []
        confidences = []
        
        # Filter by confidence - only keep good detections
        MIN_CONFIDENCE = 0.3
        
        for result in results:
            # Handle both formats: (bbox, text, confidence) or (text, confidence)
            if len(result) == 3:
                bbox, text, confidence = result
            elif len(result) == 2:
                text, confidence = result
                bbox = None
            else:
                continue
            # Skip low confidence results
            try:
                conf_float = float(confidence) if not isinstance(confidence, (int, float)) else confidence
            except (ValueError, TypeError):
                conf_float = 0.0
            if conf_float < MIN_CONFIDENCE:
                continue
                
            # Clean the text
            cleaned_text = clean_ocr_text(text)
            
            # Skip empty or very short results
            if not cleaned_text or len(cleaned_text) < 2:
                continue
            
            segment_data = {
                'text': cleaned_text,
                'confidence': float(confidence),
            }
            # Add bbox if available
            if bbox is not None:
                segment_data['bbox'] = [[int(p[0]), int(p[1])] for p in bbox]
            
            segments.append(segment_data)
            texts.append(cleaned_text)
            confidences.append(confidence)
        
        combined_text = ' '.join(texts)
        # Final cleanup
        combined_text = clean_ocr_text(combined_text)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return {
            'text': combined_text,
            'segments': segments,
            'confidence': round(avg_confidence, 4)
        }
        
    except Exception as e:
        logger.error(f"OCR error: {e}")
        import traceback
        traceback.print_exc()
        return {'text': '', 'segments': [], 'confidence': 0.0}


def analyze_image_content(image: Image.Image, candidate_labels: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Analyze image content using CLIP
    
    Args:
        image: PIL Image
        candidate_labels: Labels to classify against (optional)
    
    Returns:
        {
            'labels': List[Dict],  # Matched labels with scores
            'description': str,     # Best matching description
            'is_manipulated': bool, # Basic manipulation detection
            'image_type': str       # infographic, photo, screenshot, meme, etc.
        }
    """
    model, processor = load_clip_model()
    
    if model is None:
        return {
            'labels': [],
            'description': 'Unable to analyze',
            'is_manipulated': False,
            'image_type': 'unknown'
        }
    
    # Default candidate labels for OSINT analysis
    if candidate_labels is None:
        candidate_labels = [
            # Content types
            "a photograph",
            "a screenshot",
            "an infographic with data",
            "a meme with text",
            "a news article",
            "a chart or graph",
            "a map",
            "a document",
            "a social media post",
            # Manipulation indicators
            "a real unedited photo",
            "a digitally manipulated image",
            "an AI generated image",
            # Content categories
            "political content",
            "health or medical content",
            "scientific content",
            "entertainment content",
            "news content",
            "advertisement",
        ]
    
    try:
        device = get_device()
        
        # Process image and text
        inputs = processor(
            text=candidate_labels,
            images=image,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
        
        # Parse results
        probs_np = probs.cpu().numpy()[0]
        results = []
        for label, prob in zip(candidate_labels, probs_np):
            results.append({
                'label': label,
                'score': float(prob)
            })
        
        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # Determine image type
        type_labels = ["photograph", "screenshot", "infographic", "meme", "chart", "map", "document"]
        image_type = "photo"
        for r in results[:5]:
            for t in type_labels:
                if t in r['label'].lower():
                    image_type = t
                    break
        
        # Check for manipulation indicators
        manipulation_score = 0.0
        for r in results:
            if "manipulated" in r['label'].lower() or "AI generated" in r['label'].lower():
                manipulation_score = max(manipulation_score, r['score'])
        
        is_manipulated = manipulation_score > 0.3
        
        return {
            'labels': results[:10],  # Top 10 labels
            'description': results[0]['label'] if results else 'unknown',
            'is_manipulated': is_manipulated,
            'manipulation_score': round(manipulation_score, 4),
            'image_type': image_type
        }
        
    except Exception as e:
        logger.error(f"CLIP analysis error: {e}")
        return {
            'labels': [],
            'description': 'Analysis failed',
            'is_manipulated': False,
            'image_type': 'unknown'
        }


def analyze_image(image_source: str, extract_text: bool = True, analyze_content: bool = True) -> Dict[str, Any]:
    """
    Complete image analysis pipeline
    
    Args:
        image_source: URL, base64, or file path
        extract_text: Whether to perform OCR
        analyze_content: Whether to use CLIP analysis
    
    Returns:
        Combined analysis results
    """
    result = {
        'success': False,
        'ocr': None,
        'content': None,
        'combined_text': '',
        'image_type': 'unknown',
        'is_manipulated': False,
        'error': None
    }
    
    # Fetch image
    image = fetch_image(image_source)
    if image is None:
        result['error'] = 'Failed to fetch or decode image'
        return result
    
    result['success'] = True
    result['image_size'] = {'width': image.width, 'height': image.height}
    
    # OCR - Extract text
    if extract_text:
        ocr_result = extract_text_from_image(image)
        result['ocr'] = ocr_result
        result['combined_text'] = ocr_result['text']
    
    # CLIP - Analyze content
    if analyze_content:
        content_result = analyze_image_content(image)
        result['content'] = content_result
        result['image_type'] = content_result['image_type']
        result['is_manipulated'] = content_result['is_manipulated']
    
    return result


# Pre-load models on module import (optional, can be lazy loaded)
def preload_models():
    """Pre-load all image analysis models"""
    logger.info("Pre-loading image analysis models...")
    load_ocr_model()
    load_clip_model()
    logger.info("Image analysis models ready!")
