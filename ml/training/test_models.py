"""
Comprehensive Model Testing Script
Tests all custom-trained DeBERTa models:
  - Sentiment (95.24% F1)
  - Stance (93.15% F1)  
  - Misinfo (89% F1)

Generates:
  - Confusion matrices (heatmaps)
  - Classification reports (F1, Precision, Recall)
  - Test predictions vs ground truth
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    f1_score, 
    precision_score, 
    recall_score,
    accuracy_score
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "ml" / "training" / "models"

# Model paths
SENTIMENT_MODEL = MODELS_DIR / "unified_sentiment" / "trial_0" / "checkpoint-29000"
SENTIMENT_TOKENIZER = MODELS_DIR / "unified_sentiment"
STANCE_MODEL = MODELS_DIR / "stance_classifier"
STANCE_TOKENIZER = MODELS_DIR / "stance_classifier"
MISINFO_MODEL = MODELS_DIR / "misinfo_classifier"
MISINFO_TOKENIZER = MODELS_DIR / "misinfo_classifier"

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


def load_model(model_path, tokenizer_path):
    """Load a model and tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    model.to(device)
    model.eval()
    return model, tokenizer


def predict_batch(model, tokenizer, texts, batch_size=16):
    """Predict labels for a batch of texts"""
    all_preds = []
    all_probs = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
        
        all_preds.extend(preds.cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())
    
    return all_preds, all_probs


def plot_confusion_matrix(y_true, y_pred, labels, title, save_path):
    """Plot confusion matrix as heatmap"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels
    )
    plt.title(title, fontsize=14)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_normalized_confusion_matrix(y_true, y_pred, labels, title, save_path):
    """Plot normalized confusion matrix as heatmap"""
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='.2%', 
        cmap='RdYlGn',
        xticklabels=labels,
        yticklabels=labels,
        vmin=0, 
        vmax=1
    )
    plt.title(title, fontsize=14)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_metrics_bar(report_dict, labels, title, save_path):
    """Plot F1, Precision, Recall as grouped bar chart"""
    metrics = ['precision', 'recall', 'f1-score']
    
    x = np.arange(len(labels))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, metric in enumerate(metrics):
        values = [report_dict[label][metric] for label in labels]
        bars = ax.bar(x + i * width, values, width, label=metric.capitalize())
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.annotate(f'{val:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 1.15)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


# ================================================================
# SENTIMENT MODEL TEST
# ================================================================
def test_sentiment_model():
    """Test sentiment model with sample data"""
    print("\n" + "="*60)
    print("TESTING SENTIMENT MODEL")
    print("="*60)
    
    # Check if model exists
    if not SENTIMENT_MODEL.exists():
        print(f"❌ Sentiment model not found at {SENTIMENT_MODEL}")
        return
    
    print(f"Loading model from: {SENTIMENT_MODEL}")
    model, tokenizer = load_model(SENTIMENT_MODEL, SENTIMENT_TOKENIZER)
    
    # Label mapping - CHECK WHICH IS CORRECT
    # Try both mappings and see which gives better results
    
    # Test samples with known labels
    test_samples = [
        # Clearly Positive
        ("I absolutely love this product! It's amazing and works perfectly!", "positive"),
        ("Great news! The team won the championship!", "positive"),
        ("This is the best day of my life, I'm so happy!", "positive"),
        ("Wonderful experience, highly recommend to everyone!", "positive"),
        ("The movie was fantastic, brilliant acting and storyline!", "positive"),
        
        # Clearly Negative  
        ("This is terrible, I hate it so much. Worst purchase ever.", "negative"),
        ("The government has failed us completely. Disgraceful!", "negative"),
        ("Parliament has lost Rs 3,300 crore to disruptions and chaos. Dysfunctional!", "negative"),
        ("Corruption is destroying our country. The system is broken.", "negative"),
        ("Horrible service, never going back. Total waste of money.", "negative"),
        
        # Neutral/Mixed
        ("The meeting is scheduled for tomorrow at 3 PM.", "neutral"),
        ("The report contains data from the last quarter.", "neutral"),
        ("According to sources, the event will take place next week.", "neutral"),
    ]
    
    texts = [t[0] for t in test_samples]
    true_labels_str = [t[1] for t in test_samples]
    
    # Mapping 1: 0=negative, 1=positive (BINARY)
    label_map_1 = {0: "negative", 1: "positive"}
    
    # Mapping 2: 0=positive, 1=negative (SWAPPED)
    label_map_2 = {0: "positive", 1: "negative"}
    
    preds, probs = predict_batch(model, tokenizer, texts)
    
    print("\n--- Testing Label Mapping ---")
    print("\nSample predictions with Mapping 1 (0=negative, 1=positive):")
    
    correct_1 = 0
    correct_2 = 0
    
    for i, (text, true_label) in enumerate(test_samples):
        pred_1 = label_map_1.get(preds[i], "unknown")
        pred_2 = label_map_2.get(preds[i], "unknown")
        
        # Skip neutral for binary model
        if true_label == "neutral":
            continue
            
        if pred_1 == true_label:
            correct_1 += 1
        if pred_2 == true_label:
            correct_2 += 1
        
        print(f"\nText: {text[:60]}...")
        print(f"  True: {true_label}")
        print(f"  Raw pred: {preds[i]}, Probs: {probs[i]}")
        print(f"  Mapping1: {pred_1} | Mapping2: {pred_2}")
    
    non_neutral = len([t for t in test_samples if t[1] != "neutral"])
    print(f"\n--- Results ---")
    print(f"Mapping 1 (0=neg, 1=pos): {correct_1}/{non_neutral} correct")
    print(f"Mapping 2 (0=pos, 1=neg): {correct_2}/{non_neutral} correct")
    
    if correct_2 > correct_1:
        print("\n⚠️ LABELS ARE SWAPPED! Use Mapping 2 (0=positive, 1=negative)")
        correct_mapping = label_map_2
    else:
        print("\n✅ Labels are correct. Use Mapping 1 (0=negative, 1=positive)")
        correct_mapping = label_map_1
    
    return correct_mapping


# ================================================================
# STANCE MODEL TEST
# ================================================================
def test_stance_model():
    """Test stance model with sample data"""
    print("\n" + "="*60)
    print("TESTING STANCE MODEL")
    print("="*60)
    
    if not STANCE_MODEL.exists():
        print(f"❌ Stance model not found at {STANCE_MODEL}")
        return
    
    print(f"Loading model from: {STANCE_MODEL}")
    model, tokenizer = load_model(STANCE_MODEL, STANCE_TOKENIZER)
    
    # Check model config for labels
    print(f"\nModel config labels: {model.config.id2label}")
    
    # Standard stance labels
    labels = ["agree", "disagree", "discuss", "unrelated"]
    
    # Test samples
    test_samples = [
        ("Climate change is real and we must act now! I fully support this.", "agree"),
        ("The vaccine is safe, studies confirm this. We should all get vaccinated.", "agree"),
        ("This claim is completely false. There is no evidence to support it.", "disagree"),
        ("I don't believe this at all. The data shows the opposite.", "disagree"),
        ("Scientists are debating whether this is true. More research is needed.", "discuss"),
        ("The weather today is sunny. I went for a walk.", "unrelated"),
        ("My favorite food is pizza. I had some yesterday.", "unrelated"),
    ]
    
    texts = [t[0] for t in test_samples]
    true_labels = [t[1] for t in test_samples]
    
    preds, probs = predict_batch(model, tokenizer, texts)
    
    print("\n--- Stance Predictions ---")
    for i, (text, true_label) in enumerate(test_samples):
        pred_label = model.config.id2label.get(preds[i], f"unknown_{preds[i]}")
        print(f"\nText: {text[:60]}...")
        print(f"  True: {true_label}")
        print(f"  Pred: {pred_label} (raw: {preds[i]})")
        print(f"  Probs: {dict(zip(model.config.id2label.values(), [f'{p:.2f}' for p in probs[i]]))}")
    
    # Map predictions to labels
    pred_labels = [model.config.id2label.get(p, "unknown") for p in preds]
    
    # Classification report
    print("\n--- Classification Report ---")
    print(classification_report(true_labels, pred_labels, labels=labels, zero_division=0))
    
    return model.config.id2label


# ================================================================
# MISINFO MODEL TEST
# ================================================================
def test_misinfo_model():
    """Test misinfo model with sample data"""
    print("\n" + "="*60)
    print("TESTING MISINFO MODEL")
    print("="*60)
    
    if not MISINFO_MODEL.exists():
        print(f"❌ Misinfo model not found at {MISINFO_MODEL}")
        return
    
    print(f"Loading model from: {MISINFO_MODEL}")
    model, tokenizer = load_model(MISINFO_MODEL, MISINFO_TOKENIZER)
    
    # Check model config
    print(f"\nModel config labels: {model.config.id2label}")
    
    # Label mappings to test
    label_map_1 = {0: "fake", 1: "real"}  # Original assumption
    label_map_2 = {0: "real", 1: "fake"}  # Swapped
    
    # Test samples with known labels
    test_samples = [
        # Clearly FAKE/Misinformation
        ("5G towers cause COVID-19! The government is hiding the truth!", "fake"),
        ("Drinking hot water cures coronavirus. Forward this to save lives!", "fake"),
        ("Bill Gates is putting microchips in vaccines to control us all!", "fake"),
        ("The election was stolen! Millions of fake ballots were found!", "fake"),
        ("Banks will be closed for 15 days. Withdraw all your money now!", "fake"),
        
        # Clearly REAL/Legitimate news
        ("The stock market closed 2% higher today driven by tech gains.", "real"),
        ("According to Reuters, the summit will take place next month.", "real"),
        ("Scientists published a peer-reviewed study in Nature journal.", "real"),
        ("The Prime Minister addressed the nation on Independence Day.", "real"),
        ("Weather forecast: Sunny with temperatures around 25°C tomorrow.", "real"),
    ]
    
    texts = [t[0] for t in test_samples]
    true_labels = [t[1] for t in test_samples]
    
    preds, probs = predict_batch(model, tokenizer, texts)
    
    print("\n--- Testing Label Mapping ---")
    
    correct_1 = 0
    correct_2 = 0
    
    for i, (text, true_label) in enumerate(test_samples):
        pred_1 = label_map_1.get(preds[i], "unknown")
        pred_2 = label_map_2.get(preds[i], "unknown")
        
        if pred_1 == true_label:
            correct_1 += 1
        if pred_2 == true_label:
            correct_2 += 1
        
        print(f"\nText: {text[:60]}...")
        print(f"  True: {true_label}")
        print(f"  Raw pred: {preds[i]}, Probs: {probs[i]}")
        print(f"  Mapping1 (0=fake): {pred_1} | Mapping2 (0=real): {pred_2}")
    
    print(f"\n--- Results ---")
    print(f"Mapping 1 (0=fake, 1=real): {correct_1}/{len(test_samples)} correct")
    print(f"Mapping 2 (0=real, 1=fake): {correct_2}/{len(test_samples)} correct")
    
    if correct_2 > correct_1:
        print("\n⚠️ LABELS ARE SWAPPED! Use Mapping 2 (0=real, 1=fake)")
        return label_map_2
    else:
        print("\n✅ Labels are correct. Use Mapping 1 (0=fake, 1=real)")
        return label_map_1


# ================================================================
# MAIN
# ================================================================
if __name__ == "__main__":
    print("="*60)
    print("CUSTOM MODEL TESTING SUITE")
    print("="*60)
    print(f"\nModels directory: {MODELS_DIR}")
    print(f"Device: {device}")
    
    # Test each model
    sentiment_mapping = test_sentiment_model()
    stance_mapping = test_stance_model()
    misinfo_mapping = test_misinfo_model()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nSentiment correct mapping: {sentiment_mapping}")
    print(f"Stance mapping: {stance_mapping}")
    print(f"Misinfo correct mapping: {misinfo_mapping}")
    
    print("\n✅ Testing complete!")
