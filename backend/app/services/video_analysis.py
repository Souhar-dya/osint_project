"""
Video and GIF Analysis Module
Extracts frames, detects text, analyzes content from video/GIF media
"""
import logging
import io
import base64
import tempfile
import os
from typing import Dict, Any, Optional, List
from pathlib import Path

import torch
import requests

logger = logging.getLogger(__name__)

# Try importing video processing libraries
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV not available - video analysis will be limited")

try:
    from PIL import Image, ImageSequence
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def fetch_media(media_source: str) -> Optional[bytes]:
    """
    Fetch video/GIF from URL or decode from base64
    
    Args:
        media_source: URL string or base64 encoded data
    
    Returns:
        Raw bytes or None if failed
    """
    try:
        if media_source.startswith('data:'):
            # Base64 encoded
            header, data = media_source.split(',', 1)
            return base64.b64decode(data)
        
        elif media_source.startswith('http'):
            # URL
            response = requests.get(media_source, timeout=30, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            response.raise_for_status()
            return response.content
        
        elif Path(media_source).exists():
            # Local file path
            with open(media_source, 'rb') as f:
                return f.read()
        
        else:
            logger.error(f"Unknown media source format: {media_source[:50]}...")
            return None
            
    except Exception as e:
        logger.error(f"Failed to fetch media: {e}")
        return None


def extract_gif_frames(gif_data: bytes, max_frames: int = 10) -> List[Any]:
    """
    Extract frames from a GIF
    
    Args:
        gif_data: Raw GIF bytes
        max_frames: Maximum frames to extract (evenly distributed)
    
    Returns:
        List of PIL Images
    """
    if not PIL_AVAILABLE:
        return []
    
    try:
        gif = Image.open(io.BytesIO(gif_data))
        frames = []
        
        # Get total frame count
        try:
            total_frames = gif.n_frames
        except:
            total_frames = 1
        
        # Calculate frame interval for even distribution
        if total_frames <= max_frames:
            frame_indices = range(total_frames)
        else:
            interval = total_frames / max_frames
            frame_indices = [int(i * interval) for i in range(max_frames)]
        
        # Extract frames
        for idx in frame_indices:
            try:
                gif.seek(idx)
                frame = gif.copy().convert('RGB')
                frames.append(frame)
            except EOFError:
                break
        
        logger.info(f"Extracted {len(frames)} frames from GIF")
        return frames
        
    except Exception as e:
        logger.error(f"GIF frame extraction error: {e}")
        return []


def extract_video_frames(video_data: bytes, max_frames: int = 10) -> List[Any]:
    """
    Extract frames from a video file
    
    Args:
        video_data: Raw video bytes
        max_frames: Maximum frames to extract (evenly distributed)
    
    Returns:
        List of PIL Images
    """
    if not CV2_AVAILABLE or not PIL_AVAILABLE:
        logger.warning("OpenCV or PIL not available for video processing")
        return []
    
    frames = []
    temp_path = None
    
    try:
        # Write to temp file (OpenCV needs file path)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            tmp.write(video_data)
            temp_path = tmp.name
        
        # Open video
        cap = cv2.VideoCapture(temp_path)
        
        if not cap.isOpened():
            logger.error("Could not open video file")
            return []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        logger.info(f"Video: {total_frames} frames, {fps:.1f} FPS, {duration:.1f}s duration")
        
        # Calculate frame interval for even distribution
        if total_frames <= max_frames:
            frame_indices = range(total_frames)
        else:
            interval = total_frames / max_frames
            frame_indices = [int(i * interval) for i in range(max_frames)]
        
        # Extract frames
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret:
                # Convert BGR to RGB and to PIL Image
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                frames.append(pil_image)
        
        cap.release()
        logger.info(f"Extracted {len(frames)} frames from video")
        
    except Exception as e:
        logger.error(f"Video frame extraction error: {e}")
    
    finally:
        # Clean up temp file
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
    
    return frames


def analyze_video_or_gif(
    media_source: str,
    max_frames: int = 8
) -> Dict[str, Any]:
    """
    Analyze video or GIF content
    
    Args:
        media_source: URL, base64, or file path
        max_frames: Maximum frames to analyze
    
    Returns:
        {
            'media_type': 'video' | 'gif',
            'frame_count': int,
            'duration_seconds': float,
            'ocr_results': List[Dict],  # Text found in each frame
            'combined_text': str,        # All text combined
            'content_analysis': Dict,    # CLIP analysis of frames
            'frame_descriptions': List,  # Description of each frame
            'misinfo_signals': List[str] # Suspicious signals found
        }
    """
    from app.services.image_analysis import extract_text_from_image, analyze_image_content
    
    result = {
        'media_type': 'unknown',
        'frame_count': 0,
        'duration_seconds': 0.0,
        'ocr_results': [],
        'combined_text': '',
        'content_analysis': {},
        'frame_descriptions': [],
        'misinfo_signals': []
    }
    
    # Fetch media
    media_data = fetch_media(media_source)
    if media_data is None:
        return result
    
    # Determine media type and extract frames
    frames = []
    
    # Check for GIF magic bytes
    if media_data[:6] in (b'GIF87a', b'GIF89a'):
        result['media_type'] = 'gif'
        frames = extract_gif_frames(media_data, max_frames)
    else:
        # Assume video
        result['media_type'] = 'video'
        frames = extract_video_frames(media_data, max_frames)
    
    if not frames:
        logger.warning("No frames extracted from media")
        return result
    
    result['frame_count'] = len(frames)
    
    # Analyze each frame
    all_texts = []
    frame_labels = []
    
    for i, frame in enumerate(frames):
        try:
            # OCR - Extract text
            ocr_result = extract_text_from_image(frame)
            if ocr_result['text']:
                all_texts.append(ocr_result['text'])
                result['ocr_results'].append({
                    'frame': i,
                    'text': ocr_result['text'],
                    'confidence': ocr_result['confidence']
                })
            
            # CLIP - Analyze content (only for key frames to save time)
            if i % 3 == 0:  # Every 3rd frame
                content = analyze_image_content(frame)
                if content.get('description'):
                    result['frame_descriptions'].append({
                        'frame': i,
                        'description': content['description'],
                        'type': content.get('image_type', 'unknown')
                    })
                    frame_labels.append(content.get('image_type', ''))
        
        except Exception as e:
            logger.error(f"Frame {i} analysis error: {e}")
    
    # Combine all extracted text
    result['combined_text'] = ' '.join(all_texts)
    
    # Detect misinfo signals from video/GIF content
    signals = detect_video_misinfo_signals(
        result['combined_text'],
        frame_labels,
        result['frame_descriptions']
    )
    result['misinfo_signals'] = signals
    
    return result


def detect_video_misinfo_signals(
    text: str,
    frame_labels: List[str],
    frame_descriptions: List[Dict]
) -> List[str]:
    """
    Detect misinformation signals specific to video/GIF content
    
    Returns:
        List of detected signals
    """
    signals = []
    text_lower = text.lower()
    
    # Text overlay patterns common in misinformation videos
    misinfo_text_patterns = [
        ('breaking', 'breaking news overlay'),
        ('share before deleted', 'viral share bait'),
        ('they don\'t want you', 'conspiracy framing'),
        ('wake up', 'alarmist language'),
        ('truth revealed', 'sensational claim'),
        ('exposed', 'exposé framing'),
        ('leaked', 'leaked content claim'),
        ('banned', 'censorship claim'),
        ('proof', 'unverified proof claim'),
        ('100% real', 'authenticity claim'),
        ('not fake', 'authenticity claim'),
        ('viral', 'viral content'),
        ('must watch', 'urgency bait'),
        ('shocking', 'sensational language'),
    ]
    
    for pattern, signal in misinfo_text_patterns:
        if pattern in text_lower:
            if signal not in signals:
                signals.append(signal)
    
    # Check frame types
    meme_count = sum(1 for label in frame_labels if 'meme' in label.lower())
    screenshot_count = sum(1 for label in frame_labels if 'screenshot' in label.lower())
    
    if meme_count > len(frame_labels) * 0.3:
        signals.append('meme-style content')
    
    if screenshot_count > len(frame_labels) * 0.5:
        signals.append('screenshot compilation')
    
    # Heavy text overlay (common in misinfo videos)
    if len(text) > 200:
        signals.append('heavy text overlay')
    
    return signals


def get_video_misinfo_boost(video_analysis: Dict[str, Any]) -> tuple:
    """
    Calculate misinfo score boost based on video analysis
    
    Returns:
        (score_boost, triggers)
    """
    boost = 0.0
    triggers = []
    
    signals = video_analysis.get('misinfo_signals', [])
    
    # Weight different signals
    signal_weights = {
        'breaking news overlay': 0.08,
        'viral share bait': 0.12,
        'conspiracy framing': 0.15,
        'alarmist language': 0.08,
        'sensational claim': 0.10,
        'exposé framing': 0.08,
        'leaked content claim': 0.10,
        'censorship claim': 0.10,
        'unverified proof claim': 0.12,
        'authenticity claim': 0.10,
        'viral content': 0.05,
        'urgency bait': 0.08,
        'sensational language': 0.08,
        'meme-style content': 0.10,
        'screenshot compilation': 0.08,
        'heavy text overlay': 0.05,
    }
    
    for signal in signals:
        if signal in signal_weights:
            boost += signal_weights[signal]
            triggers.append(f"video: {signal}")
    
    return min(boost, 0.30), triggers  # Cap at 0.30
