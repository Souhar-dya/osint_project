"""
Comprehensive Model Evaluation Script
Generates confusion matrices with heatmaps and detailed metrics
for Sentiment, Stance, and Misinfo models
Compares custom models vs pretrained baselines
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    f1_score, precision_score, recall_score, accuracy_score
)
from datasets import load_dataset
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "evaluation_results"
OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Base tokenizer for all DeBERTa models
BASE_MODEL = "microsoft/deberta-v3-base"


def load_model(model_path, num_labels=2):
    """Load model and tokenizer"""
    # Use base tokenizer since checkpoints may not have tokenizer saved
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(DEVICE)
    model.eval()
    return model, tokenizer


def predict_batch(model, tokenizer, texts, batch_size=16):
    """Predict on a batch of texts"""
    all_preds = []
    all_probs = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
        
        all_preds.extend(preds.cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())
    
    return all_preds, all_probs


def plot_confusion_matrix(y_true, y_pred, labels, title, output_path):
    """Plot and save confusion matrix heatmap"""
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        annot_kws={"size": 14}
    )
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")
    
    return cm


def plot_normalized_confusion_matrix(y_true, y_pred, labels, title, output_path):
    """Plot normalized confusion matrix (percentages)"""
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)), normalize='true')
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='.2%', 
        cmap='RdYlGn',
        xticklabels=labels,
        yticklabels=labels,
        annot_kws={"size": 12},
        vmin=0, 
        vmax=1
    )
    plt.title(f"{title} (Normalized)", fontsize=16, fontweight='bold')
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def print_metrics(y_true, y_pred, labels, model_name):
    """Print detailed metrics"""
    print(f"\n{'='*60}")
    print(f"{model_name} - Detailed Metrics")
    print('='*60)
    
    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:         {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  F1 (macro):       {f1_macro:.4f}")
    print(f"  F1 (weighted):    {f1_weighted:.4f}")
    print(f"  Precision (macro): {precision_macro:.4f}")
    print(f"  Recall (macro):    {recall_macro:.4f}")
    
    # Per-class metrics
    print(f"\nPer-Class Metrics:")
    report = classification_report(y_true, y_pred, target_names=labels, zero_division=0)
    print(report)
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro
    }


# ============================================================
# BASELINE COMPARISONS
# ============================================================

def evaluate_zero_shot_stance(test_data, labels):
    """Evaluate zero-shot BART for stance detection"""
    print("\n  Evaluating Zero-Shot BART (facebook/bart-large-mnli)...")
    
    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=0 if DEVICE == "cuda" else -1
    )
    
    candidate_labels = ['agree', 'disagree', 'discuss', 'unrelated']
    label_map = {l: i for i, l in enumerate(candidate_labels)}
    
    preds = []
    for h, b, _ in test_data:
        text = f"Headline: {h}\nBody: {b}"
        result = classifier(text, candidate_labels)
        pred_label = result['labels'][0]
        preds.append(label_map[pred_label])
    
    true_labels = [l for _, _, l in test_data]
    return preds, true_labels


def evaluate_pretrained_misinfo(test_data, custom_labels):
    """Evaluate pretrained fake news detector vs custom model"""
    print("\n  Evaluating Pretrained Fake News Detector...")
    
    # Try a pretrained fake news model
    pretrained_models = [
        "jy46604790/Fake-News-Bert-Detect",
        "hamzab/roberta-fake-news-classification",
    ]
    
    results = {}
    
    for model_name in pretrained_models:
        try:
            print(f"    Trying: {model_name}")
            classifier = pipeline(
                "text-classification",
                model=model_name,
                device=0 if DEVICE == "cuda" else -1
            )
            
            texts = [t for t, _ in test_data]
            true_labels = [l for _, l in test_data]
            
            preds = []
            for text in texts:
                try:
                    result = classifier(text[:512], truncation=True)
                    label = result[0]['label'].lower()
                    # Map various label formats to 0 (fake) or 1 (real)
                    if 'fake' in label or 'false' in label or label == '0' or label == 'label_0':
                        preds.append(0)
                    else:
                        preds.append(1)
                except Exception as e:
                    preds.append(1)  # Default to real on error
            
            acc = accuracy_score(true_labels, preds)
            f1 = f1_score(true_labels, preds, average='macro', zero_division=0)
            results[model_name] = {'accuracy': acc, 'f1': f1, 'preds': preds}
            print(f"      Accuracy: {acc*100:.2f}%, F1: {f1:.4f}")
            
        except Exception as e:
            print(f"      Failed to load: {e}")
            continue
    
    return results


def compare_custom_vs_baseline(custom_metrics, baseline_metrics, model_name):
    """Create comparison visualization"""
    if not baseline_metrics:
        return
        
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = [f'Custom {model_name}'] + list(baseline_metrics.keys())
    accuracies = [custom_metrics['accuracy']] + [m['accuracy'] for m in baseline_metrics.values()]
    f1_scores = [custom_metrics['f1_macro']] + [m['f1'] for m in baseline_metrics.values()]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', color='#3498db')
    bars2 = ax.bar(x + width/2, f1_scores, width, label='F1 Score', color='#2ecc71')
    
    # Add value labels
    for bar, val in zip(bars1, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
               f'{val*100:.1f}%', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, f1_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
               f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'{model_name} - Custom vs Pretrained Baseline Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.split('/')[-1][:20] for m in models], rotation=15, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / f"{model_name.lower()}_vs_baseline.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved comparison: {output_path}")


# ============================================================
# SENTIMENT EVALUATION
# ============================================================
def evaluate_sentiment():
    print("\n" + "="*60)
    print("EVALUATING SENTIMENT MODEL")
    print("="*60)
    
    model_path = MODELS_DIR / "unified_sentiment" / "trial_0" / "checkpoint-29000"
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        return None
    
    model, tokenizer = load_model(model_path)
    labels = ['negative', 'positive']
    
    # Load test data - SST2 from GLUE
    print("Loading SST-2 test data...")
    try:
        dataset = load_dataset("glue", "sst2", split="validation")
        texts = dataset['sentence'][:500]  # Use 500 samples
        true_labels = dataset['label'][:500]  # SST2: 0=negative, 1=positive
        print(f"  Loaded {len(texts)} samples")
    except Exception as e:
        print(f"  Error loading dataset: {e}")
        # Use manual test set
        texts = [
            "This movie is absolutely terrible and boring",
            "I love this product, it's amazing!",
            "Worst experience ever, never going back",
            "Fantastic service, highly recommend",
            "The food was disgusting and cold",
            "Beautiful scenery and wonderful atmosphere",
            "Complete waste of money",
            "Best purchase I've ever made",
            "Horrible customer service",
            "Incredibly happy with this"
        ] * 10
        true_labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 10
        print(f"  Using manual test set: {len(texts)} samples")
    
    # Predict
    print("Running predictions...")
    preds, probs = predict_batch(model, tokenizer, texts)
    
    # Metrics
    metrics = print_metrics(true_labels, preds, labels, "SENTIMENT")
    
    # Confusion matrices
    print("\nGenerating confusion matrices...")
    plot_confusion_matrix(
        true_labels, preds, labels,
        "Sentiment Model - Confusion Matrix",
        OUTPUT_DIR / "sentiment_confusion_matrix.png"
    )
    plot_normalized_confusion_matrix(
        true_labels, preds, labels,
        "Sentiment Model - Confusion Matrix",
        OUTPUT_DIR / "sentiment_confusion_matrix_normalized.png"
    )
    
    return metrics


# ============================================================
# STANCE EVALUATION  
# ============================================================
def evaluate_stance():
    print("\n" + "="*60)
    print("EVALUATING STANCE MODEL")
    print("="*60)
    
    # Use checkpoint-939 (best checkpoint)
    model_path = MODELS_DIR / "stance_classifier" / "checkpoint-939"
    if not model_path.exists():
        # Fallback to root
        model_path = MODELS_DIR / "stance_classifier"
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        return None
    
    model, tokenizer = load_model(model_path, num_labels=4)
    labels = ['agree', 'disagree', 'discuss', 'unrelated']
    
    # Load FNC-1 style test data
    print("Creating stance test data...")
    # Manual test data (FNC-1 style: headline + body)
    test_data = [
        # Agree examples
        ("Scientists confirm climate change is accelerating", "New research published today confirms that climate change is happening faster than predicted. Multiple studies agree on this finding.", 0),
        ("Vaccine proven safe in trials", "Clinical trials have demonstrated the vaccine's safety profile. Researchers agree the benefits outweigh risks.", 0),
        ("Economy shows strong growth", "Economic indicators confirm robust growth this quarter. Experts agree the trend will continue.", 0),
        
        # Disagree examples
        ("Earth is flat according to experts", "Scientists worldwide have definitively proven the Earth is spherical. There is no credible evidence supporting flat earth claims.", 1),
        ("5G causes health problems", "Multiple peer-reviewed studies have found no link between 5G and health issues. The claims are unfounded.", 1),
        ("Vaccines contain microchips", "This claim has been thoroughly debunked. Vaccines contain no tracking devices or microchips.", 1),
        
        # Discuss examples  
        ("New policy may impact economy", "Economists are divided on the effects of the new policy. Some predict growth while others warn of potential risks.", 2),
        ("Study findings spark debate", "The controversial study has generated discussion among researchers. More investigation is needed.", 2),
        ("Experts analyze election results", "Political analysts are examining the election outcomes and discussing various interpretations of the data.", 2),
        
        # Unrelated examples
        ("Weather forecast for tomorrow", "The local football team won their match 3-1 yesterday. Fans celebrated throughout the night.", 3),
        ("Tech company releases new phone", "The recipe for chocolate cake requires flour, sugar, eggs, and cocoa powder. Mix well and bake.", 3),
        ("President visits foreign country", "My cat likes to sleep on the windowsill in the afternoon sun. She's very cute.", 3),
    ]
    
    texts = [f"{h} [SEP] {b}" for h, b, _ in test_data]
    true_labels = [l for _, _, l in test_data]
    print(f"  Created {len(texts)} test samples")
    
    # Predict
    print("Running predictions...")
    preds, probs = predict_batch(model, tokenizer, texts)
    
    # Metrics
    metrics = print_metrics(true_labels, preds, labels, "STANCE (Custom DeBERTa)")
    
    # Confusion matrices
    print("\nGenerating confusion matrices...")
    plot_confusion_matrix(
        true_labels, preds, labels,
        "Stance Model (Custom) - Confusion Matrix",
        OUTPUT_DIR / "stance_confusion_matrix.png"
    )
    plot_normalized_confusion_matrix(
        true_labels, preds, labels,
        "Stance Model (Custom) - Confusion Matrix",
        OUTPUT_DIR / "stance_confusion_matrix_normalized.png"
    )
    
    # === COMPARE WITH ZERO-SHOT BART ===
    print("\n" + "-"*40)
    print("COMPARING WITH ZERO-SHOT BART BASELINE")
    print("-"*40)
    
    try:
        zs_preds, zs_true = evaluate_zero_shot_stance(test_data, labels)
        zs_metrics = print_metrics(zs_true, zs_preds, labels, "STANCE (Zero-Shot BART)")
        
        # Zero-shot confusion matrix
        plot_confusion_matrix(
            zs_true, zs_preds, labels,
            "Stance Model (Zero-Shot BART) - Confusion Matrix",
            OUTPUT_DIR / "stance_zeroshot_confusion_matrix.png"
        )
        
        # Create comparison chart
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(2)
        width = 0.35
        
        custom_vals = [metrics['accuracy'], metrics['f1_macro']]
        zs_vals = [zs_metrics['accuracy'], zs_metrics['f1_macro']]
        
        bars1 = ax.bar(x - width/2, custom_vals, width, label='Custom DeBERTa', color='#e74c3c')
        bars2 = ax.bar(x + width/2, zs_vals, width, label='Zero-Shot BART', color='#3498db')
        
        for bar, val in zip(bars1, custom_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                   f'{val*100:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
        for bar, val in zip(bars2, zs_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                   f'{val*100:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Stance Detection: Custom DeBERTa vs Zero-Shot BART', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['Accuracy', 'F1 Score (Macro)'])
        ax.legend()
        ax.set_ylim(0, 1.2)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "stance_custom_vs_zeroshot.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved comparison: {OUTPUT_DIR / 'stance_custom_vs_zeroshot.png'}")
        
        # Verdict
        print("\n  📊 STANCE COMPARISON RESULTS:")
        print(f"     Custom DeBERTa:  {metrics['accuracy']*100:.2f}% accuracy, {metrics['f1_macro']:.4f} F1")
        print(f"     Zero-Shot BART:  {zs_metrics['accuracy']*100:.2f}% accuracy, {zs_metrics['f1_macro']:.4f} F1")
        if zs_metrics['accuracy'] > metrics['accuracy']:
            print("     ✅ VERDICT: Zero-Shot BART performs BETTER - recommended for production")
        else:
            print("     ✅ VERDICT: Custom DeBERTa performs BETTER - use custom model")
            
    except Exception as e:
        print(f"  Error comparing with zero-shot: {e}")
    
    return metrics


# ============================================================
# MISINFO EVALUATION
# ============================================================
def evaluate_misinfo():
    print("\n" + "="*60)
    print("EVALUATING MISINFO MODEL")
    print("="*60)
    
    # Use checkpoint-1692 (best checkpoint - 80%)
    model_path = MODELS_DIR / "misinfo_classifier" / "checkpoint-1692"
    if not model_path.exists():
        model_path = MODELS_DIR / "misinfo_classifier"
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        return None
    
    model, tokenizer = load_model(model_path)
    labels = ['fake', 'real']
    
    # Create test data
    print("Creating misinfo test data...")
    test_data = [
        # Fake news examples
        ("5G towers spread coronavirus and cause cancer in humans", 0),
        ("Bill Gates is implanting microchips through COVID vaccines", 0),
        ("Drinking bleach cures COVID-19 according to doctors", 0),
        ("The moon landing was faked by NASA in a Hollywood studio", 0),
        ("Election was stolen with millions of fake ballots", 0),
        ("Banks will close for 30 days withdraw all money now", 0),
        ("Forwarding this message will give you free mobile data", 0),
        ("Eating garlic cures coronavirus confirmed by WHO", 0),
        ("Government hiding alien contact from public", 0),
        ("Chemtrails are poisoning the population secretly", 0),
        ("COVID vaccine changes your DNA permanently", 0),
        ("Flat earth society proves NASA lies about globe", 0),
        
        # Real news examples
        ("Stock market closed 2% higher driven by tech sector gains", 1),
        ("Scientists publish peer-reviewed study in Nature journal", 1),
        ("Prime Minister addressed nation on Independence Day", 1),
        ("Weather forecast shows temperatures around 25 degrees tomorrow", 1),
        ("Reuters reports summit scheduled for next month", 1),
        ("Central bank announces interest rate decision", 1),
        ("University research team discovers new treatment method", 1),
        ("Olympic committee confirms venue for next games", 1),
        ("Parliament passes new education reform bill", 1),
        ("Annual report shows company revenue increased by 15%", 1),
        ("Medical journal publishes clinical trial results", 1),
        ("Government statistics bureau releases employment data", 1),
    ]
    
    texts = [t for t, _ in test_data]
    true_labels = [l for _, l in test_data]
    print(f"  Created {len(texts)} test samples")
    
    # Predict
    print("Running predictions...")
    preds, probs = predict_batch(model, tokenizer, texts)
    
    # Metrics
    metrics = print_metrics(true_labels, preds, labels, "MISINFO (Custom DeBERTa checkpoint-1692)")
    
    # Confusion matrices
    print("\nGenerating confusion matrices...")
    plot_confusion_matrix(
        true_labels, preds, labels,
        "Misinformation Model (Custom) - Confusion Matrix",
        OUTPUT_DIR / "misinfo_confusion_matrix.png"
    )
    plot_normalized_confusion_matrix(
        true_labels, preds, labels,
        "Misinformation Model (Custom) - Confusion Matrix",
        OUTPUT_DIR / "misinfo_confusion_matrix_normalized.png"
    )
    
    # === COMPARE WITH PRETRAINED MODELS ===
    print("\n" + "-"*40)
    print("COMPARING WITH PRETRAINED BASELINES")
    print("-"*40)
    
    baseline_results = evaluate_pretrained_misinfo(test_data, labels)
    
    if baseline_results:
        # Create comparison visualization
        compare_custom_vs_baseline(metrics, baseline_results, "Misinfo")
        
        # Print verdict
        print("\n  📊 MISINFO COMPARISON RESULTS:")
        print(f"     Custom DeBERTa (ckpt-1692):  {metrics['accuracy']*100:.2f}% accuracy, {metrics['f1_macro']:.4f} F1")
        
        best_baseline_acc = 0
        best_baseline_name = ""
        for name, result in baseline_results.items():
            print(f"     {name.split('/')[-1]}: {result['accuracy']*100:.2f}% accuracy, {result['f1']:.4f} F1")
            if result['accuracy'] > best_baseline_acc:
                best_baseline_acc = result['accuracy']
                best_baseline_name = name
        
        if metrics['accuracy'] > best_baseline_acc:
            print(f"     ✅ VERDICT: Custom model WINS by {(metrics['accuracy'] - best_baseline_acc)*100:.1f}%!")
        else:
            print(f"     ⚠️ VERDICT: {best_baseline_name.split('/')[-1]} is better by {(best_baseline_acc - metrics['accuracy'])*100:.1f}%")
    
    return metrics


# ============================================================
# SUMMARY VISUALIZATION
# ============================================================
def create_summary_plot(all_metrics):
    """Create summary comparison plot"""
    if not all_metrics:
        return
    
    models = list(all_metrics.keys())
    metrics_names = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
    
    x = np.arange(len(models))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
    
    for i, metric in enumerate(metrics_names):
        values = [all_metrics[m].get(metric, 0) for m in models]
        bars = ax.bar(x + i*width, values, width, label=metric.replace('_', ' ').title(), color=colors[i])
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {OUTPUT_DIR / 'model_comparison.png'}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("="*60)
    print("MODEL EVALUATION SUITE")
    print("="*60)
    print(f"Output directory: {OUTPUT_DIR}")
    
    all_metrics = {}
    
    # Evaluate each model
    sentiment_metrics = evaluate_sentiment()
    if sentiment_metrics:
        all_metrics['Sentiment'] = sentiment_metrics
    
    stance_metrics = evaluate_stance()
    if stance_metrics:
        all_metrics['Stance'] = stance_metrics
    
    misinfo_metrics = evaluate_misinfo()
    if misinfo_metrics:
        all_metrics['Misinfo'] = misinfo_metrics
    
    # Create summary
    create_summary_plot(all_metrics)
    
    # Final summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    for model, metrics in all_metrics.items():
        print(f"\n{model}:")
        print(f"  Accuracy: {metrics['accuracy']*100:.2f}%")
        print(f"  F1 Score: {metrics['f1_macro']*100:.2f}%")
        print(f"  Precision: {metrics['precision_macro']*100:.2f}%")
        print(f"  Recall: {metrics['recall_macro']*100:.2f}%")
    
    print(f"\n✅ All results saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    for f in OUTPUT_DIR.glob("*.png"):
        print(f"  - {f.name}")
