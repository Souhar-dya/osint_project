"""
Framing & Propaganda Detection Module
Uses our trained dual-head BERT model (87% F1)
Detects media framing and propaganda techniques
"""
from typing import List
import logging
import re
import torch

from app.models.schemas import FramingResult
from app.services.model_loader import get_model, get_tokenizer, get_labels, get_device_name

logger = logging.getLogger(__name__)

# Media Frames (from Media Frames Corpus)
MEDIA_FRAMES = [
    "Economic",           # Focus on financial costs/benefits
    "Capacity and Resources",  # Availability of resources
    "Morality",          # Religious/ethical implications
    "Fairness and Equality",   # Balance of treatment
    "Legality and Constitutionality",  # Legal aspects
    "Policy Prescription",  # Specific policy recommendations
    "Crime and Punishment",  # Criminal justice frame
    "Security and Defense",  # National security focus
    "Health and Safety",  # Public health implications
    "Quality of Life",    # Impact on living standards
    "Cultural Identity",  # Cultural/national identity
    "Public Opinion",     # What people think
    "Political",          # Political implications
    "External Regulation"  # Role of external bodies
]

# Propaganda Techniques (from SemEval-2020 Task 11)
PROPAGANDA_TECHNIQUES = {
    "loaded_language": {
        "patterns": ["shocking", "outrageous", "horrific", "devastating", "explosive", 
                    "bombshell", "slam", "destroy", "crush", "annihilate"],
        "description": "Using emotionally charged words"
    },
    "name_calling": {
        "patterns": ["liar", "fraud", "fake", "corrupt", "evil", "radical", "extremist",
                    "traitor", "puppet", "shill"],
        "description": "Labeling opponents with negative names"
    },
    "exaggeration": {
        "patterns": ["always", "never", "everyone", "nobody", "all", "none", "worst ever",
                    "best ever", "unprecedented", "historic"],
        "description": "Overstating claims"
    },
    "fear_appeal": {
        "patterns": ["danger", "threat", "crisis", "emergency", "catastrophe", "disaster",
                    "collapse", "destroy", "warn", "alarming"],
        "description": "Appealing to fear"
    },
    "flag_waving": {
        "patterns": ["patriot", "freedom", "liberty", "nation", "country", "american",
                    "flag", "values", "heritage", "tradition"],
        "description": "Playing on national pride"
    },
    "doubt": {
        "patterns": ["some say", "questions remain", "allegedly", "supposedly", "rumor",
                    "unconfirmed", "sources say", "they claim"],
        "description": "Casting doubt without evidence"
    },
    "appeal_to_authority": {
        "patterns": ["experts say", "studies show", "scientists confirm", "doctors agree",
                    "research proves", "according to experts"],
        "description": "Claiming authority without specifics"
    },
    "whataboutism": {
        "patterns": ["what about", "but they", "hypocrite", "both sides", "you also"],
        "description": "Deflecting by pointing to others"
    },
    "false_dilemma": {
        "patterns": ["either", "or else", "only two choices", "must choose", "no alternative"],
        "description": "Presenting false binary choices"
    },
    "repetition": {
        "patterns": [],  # Detected by repeated phrases
        "description": "Repeating ideas for emphasis"
    }
}


def detect_framing(text: str) -> FramingResult:
    """
    Detect media framing and propaganda techniques in text.
    Uses our trained dual-head BERT model (87% F1).
    """
    try:
        model = get_model("framing")
        tokenizer = get_tokenizer("framing")
        frame_labels = get_labels("frame")
        propaganda_labels = get_labels("propaganda")
        device = get_device_name()
        
        if model is None or tokenizer is None:
            # Fallback to rule-based
            frame = _rule_based_frame(text)
            propaganda = _detect_propaganda_rules(text)
            return FramingResult(
                frame=frame,
                propaganda_techniques=propaganda,
                confidence=0.5
            )
        
        # Tokenize and run inference
        inputs = tokenizer(
            text[:512],
            return_tensors="pt",
            truncation=True,
            max_length=256
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            frame_probs = torch.sigmoid(outputs['frame_logits'])[0]
            propaganda_probs = torch.sigmoid(outputs['propaganda_logits'])[0]
        
        # Get top frame (highest probability)
        top_frame_idx = torch.argmax(frame_probs).item()
        top_frame_conf = frame_probs[top_frame_idx].item()
        frame = frame_labels.get(top_frame_idx, "Political")
        
        # Get propaganda techniques (above threshold)
        propaganda_threshold = 0.4
        detected_propaganda = []
        for idx, prob in enumerate(propaganda_probs):
            if prob.item() > propaganda_threshold:
                technique = propaganda_labels.get(idx, f"Technique_{idx}")
                detected_propaganda.append(technique)
        
        # Calculate overall confidence (average of top scores)
        confidence = top_frame_conf
        
        return FramingResult(
            frame=frame,
            propaganda_techniques=detected_propaganda[:5],  # Limit to 5
            confidence=round(confidence, 4)
        )
        
    except Exception as e:
        logger.error(f"Framing detection error: {e}")
        return FramingResult(
            frame="Unknown",
            propaganda_techniques=[],
            confidence=0.0
        )


def _rule_based_frame(text: str) -> str:
    """Fallback rule-based frame detection"""
    text_lower = text.lower()
    
    frame_keywords = {
        "Economic": ["economy", "money", "cost", "budget", "financial", "tax", "jobs", "market"],
        "Political": ["government", "policy", "vote", "election", "congress", "senate", "law"],
        "Health": ["health", "medical", "hospital", "doctor", "disease", "pandemic", "vaccine"],
        "Security": ["security", "military", "defense", "terror", "threat", "war", "attack"],
        "Morality": ["moral", "ethical", "right", "wrong", "values", "religious", "faith"],
        "Fairness": ["fair", "equal", "justice", "rights", "discrimination", "inequality"],
        "Legality": ["legal", "law", "court", "judge", "constitution", "lawsuit", "crime"],
        "Crime": ["crime", "criminal", "police", "arrest", "murder", "theft", "fraud"]
    }
    
    for frame, keywords in frame_keywords.items():
        if any(kw in text_lower for kw in keywords):
            return frame
    
    return "Political"  # Default


def _detect_propaganda_rules(text: str) -> List[str]:
    """Fallback rule-based propaganda detection"""
    detected = []
    text_lower = text.lower()
    
    for technique, data in PROPAGANDA_TECHNIQUES.items():
        if any(pattern in text_lower for pattern in data["patterns"]):
            detected.append(technique.replace("_", " ").title())
    
    return detected[:5]


def _detect_frame(text: str) -> str:
    """Legacy function - redirects to rule-based"""
    return _rule_based_frame(text)


def _detect_propaganda(text: str) -> List[str]:
    """Legacy function - redirects to rule-based"""
    return _detect_propaganda_rules(text)
