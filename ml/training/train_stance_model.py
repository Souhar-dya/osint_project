#!/usr/bin/env python3
"""
Stance Detection Model Training
- Model: DeBERTa-v3-base
- Datasets: FNC-1, MNLI (for NLI-style stance)
- Classes: agree, disagree, discuss, unrelated
- Fast Optuna tuning (3 trials)
"""

import os
import re
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import optuna

# ==================== CONFIGURATION ====================
MODEL_NAME = "microsoft/deberta-v3-base"
OUTPUT_DIR = "models/stance_classifier"
MAX_LEN = 192  # Balanced for headline + body context
NUM_LABELS = 4  # agree, disagree, discuss, unrelated
METRIC = "f1"
OPTUNA_TRIALS = 3  # Fast training
SEED = 42
DATA_DIR = "../../data"
MAX_SAMPLES = 25000  # Good balance of speed and accuracy
# =======================================================

LABEL_MAP = {"agree": 0, "disagree": 1, "discuss": 2, "unrelated": 3}
ID2LABEL = {v: k for k, v in LABEL_MAP.items()}


def get_device():
    """Detect and return CUDA device. Exits if not available."""
    if not torch.cuda.is_available():
        print("[ERROR] CUDA is not available. This script requires GPU.")
        print("[ERROR] Please ensure you have a CUDA-capable GPU and proper drivers installed.")
        exit(1)
    
    # Force CUDA device
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    
    # Clear memory
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Enable memory efficient settings
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"[INFO] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"[INFO] Free Memory: {torch.cuda.mem_get_info()[0] / 1e9:.2f} GB")
    
    return device


def normalize(text: str) -> str:
    """Clean and normalize text."""
    if pd.isna(text):
        return ""
    text = str(text).lower().strip()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()[:1000]  # Truncate long texts


def load_fnc_dataset():
    """Load FNC-1 Fake News Challenge dataset."""
    print("\n[INFO] Loading FNC-1 dataset...")
    
    stances_path = os.path.join(DATA_DIR, "fnc1", "train_stances.csv")
    bodies_path = os.path.join(DATA_DIR, "fnc1", "train_bodies.csv")
    
    stances = pd.read_csv(stances_path)
    bodies = pd.read_csv(bodies_path)
    
    # Merge headline with body
    merged = stances.merge(bodies, on="Body ID")
    
    # Create text pairs (headline + body snippet)
    merged["text"] = merged.apply(
        lambda x: f"Headline: {normalize(x['Headline'])} Article: {normalize(x['articleBody'])[:500]}", 
        axis=1
    )
    merged["label"] = merged["Stance"].map(LABEL_MAP)
    
    print(f"  ✓ FNC-1: {len(merged)} samples")
    print(f"  Labels: {merged['label'].value_counts().to_dict()}")
    
    return merged[["text", "label"]]


def load_mnli_for_stance():
    """Load MNLI and convert to stance-like labels."""
    print("\n[INFO] Loading MNLI dataset...")
    
    try:
        from datasets import load_dataset
        ds = load_dataset("multi_nli", split="train")
        
        # Sample smaller subset for faster training
        ds = ds.shuffle(seed=SEED).select(range(min(15000, len(ds))))
        df = ds.to_pandas()
        
        # Map: entailment=agree, contradiction=disagree, neutral=discuss
        label_map = {0: 0, 1: 2, 2: 1}  # entail->agree, neutral->discuss, contradict->disagree
        
        df["text"] = df.apply(
            lambda x: f"Headline: {normalize(x['premise'])} Article: {normalize(x['hypothesis'])}", 
            axis=1
        )
        df["label"] = df["label"].map(label_map)
        
        print(f"  ✓ MNLI: {len(df)} samples")
        return df[["text", "label"]]
        
    except Exception as e:
        print(f"  ✗ MNLI failed: {e}")
        return pd.DataFrame(columns=["text", "label"])


class StanceDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for stance classification."""
    
    def __init__(self, texts, labels, tokenizer, max_len=MAX_LEN):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors=None,
        )
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            "input_ids": torch.tensor(self.encodings["input_ids"][idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.encodings["attention_mask"][idx], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }
        if "token_type_ids" in self.encodings:
            item["token_type_ids"] = torch.tensor(self.encodings["token_type_ids"][idx], dtype=torch.long)
        return item


class WeightedLossTrainer(Trainer):
    """Custom Trainer with class-weighted loss."""
    
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        loss_fn = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device)
        )
        loss = loss_fn(logits, labels)
        
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    """Compute accuracy and F1 score."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    report = classification_report(labels, preds, output_dict=True, zero_division=0)
    return {
        "accuracy": report["accuracy"],
        "f1": report["weighted avg"]["f1-score"],
    }


def run_optuna_search(train_ds, val_ds, class_weights, n_trials=OPTUNA_TRIALS):
    """Run fast hyperparameter optimization with Optuna."""
    
    def objective(trial):
        lr = trial.suggest_float("lr", 1e-5, 3e-5, log=True)
        batch_size = 16  # Safe for 4GB VRAM
        epochs = 2  # Fixed for fast trials
        weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True)

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=NUM_LABELS,
            id2label=ID2LABEL, label2id=LABEL_MAP
        )
        model.cuda()  # Explicit CUDA placement

        training_args = TrainingArguments(
            output_dir=f"{OUTPUT_DIR}/trial_{trial.number}",
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=weight_decay,
            warmup_ratio=0.1,
            gradient_accumulation_steps=4,  # Effective batch = 64
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="steps",
            logging_steps=50,
            load_best_model_at_end=True,
            metric_for_best_model=METRIC,
            report_to="none",
            fp16=True,  # CUDA required
            dataloader_num_workers=0,
            dataloader_pin_memory=True,  # CUDA required
            optim="adamw_torch_fused",  # Faster optimizer
        )

        trainer = WeightedLossTrainer(
            class_weights=class_weights,
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        metrics = trainer.evaluate()
        
        del model, trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return metrics.get("eval_f1", 0.0)

    print(f"\n[INFO] Starting Optuna search ({n_trials} trials)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"\n[INFO] Best trial: {study.best_trial.number}")
    print(f"[INFO] Best F1: {study.best_value:.4f}")
    print(f"[INFO] Best params: {study.best_params}")
    
    return study.best_params


def train_final_model(train_ds, val_ds, test_ds, class_weights, best_params, test_df):
    """Train final model with best hyperparameters."""
    
    print("\n[INFO] Training final model...")
    
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS,
        id2label=ID2LABEL, label2id=LABEL_MAP
    )
    model.cuda()  # Explicit CUDA placement

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=best_params["lr"],
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,  # Slightly more for final
        weight_decay=best_params["weight_decay"],
        warmup_ratio=0.1,
        gradient_accumulation_steps=4,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model=METRIC,
        report_to="none",
        fp16=True,  # CUDA required
        dataloader_num_workers=0,
        dataloader_pin_memory=True,  # CUDA required
        optim="adamw_torch_fused",  # Faster optimizer
    )

    trainer = WeightedLossTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Evaluate
    print("\n[INFO] Evaluating on test set...")
    predictions = trainer.predict(test_ds)
    y_true = predictions.label_ids
    y_pred = np.argmax(predictions.predictions, axis=1)

    print("\n" + "=" * 50)
    print("TEST RESULTS")
    print("=" * 50)
    print(classification_report(y_true, y_pred, target_names=list(LABEL_MAP.keys())))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    trainer.save_model(OUTPUT_DIR)
    print(f"[INFO] Model saved to: {OUTPUT_DIR}")

    return trainer, model


def main():
    """Main training pipeline."""
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = get_device()

    # Load datasets
    fnc_df = load_fnc_dataset()
    mnli_df = load_mnli_for_stance()
    
    # Combine
    df = pd.concat([fnc_df, mnli_df], ignore_index=True)
    df = df.dropna()
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    
    # Limit samples for faster training
    if len(df) > MAX_SAMPLES:
        df = df.sample(n=MAX_SAMPLES, random_state=SEED).reset_index(drop=True)
        print(f"[INFO] Limited to {MAX_SAMPLES} samples for faster training")
    
    print(f"\n[INFO] Total samples: {len(df)}")
    print(f"[INFO] Class distribution:\n{df['label'].value_counts()}")

    # Split
    train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=SEED)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=SEED)
    
    print(f"\n[INFO] Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Tokenizer
    print("\n[INFO] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Create datasets
    print("[INFO] Tokenizing...")
    train_ds = StanceDataset(train_df["text"], train_df["label"], tokenizer)
    val_ds = StanceDataset(val_df["text"], val_df["label"], tokenizer)
    test_ds = StanceDataset(test_df["text"], test_df["label"], tokenizer)

    # Class weights
    classes = np.unique(train_df["label"])
    weights = compute_class_weight("balanced", classes=classes, y=train_df["label"])
    class_weights = torch.tensor(weights, dtype=torch.float)
    print(f"[INFO] Class weights: {dict(zip(classes.tolist(), weights.tolist()))}")

    # Optuna search
    best_params = run_optuna_search(train_ds, val_ds, class_weights)

    # Train final
    train_final_model(train_ds, val_ds, test_ds, class_weights, best_params, test_df)

    print("\n[INFO] Stance model training complete!")


if __name__ == "__main__":
    main()
