#!/usr/bin/env python3
"""
Misinformation Detection Model Training
- Model: DeBERTa-v3-base
- Datasets: LIAR, CoAID, FakeNewsNet, ISOT
- Classes: fake, real (binary)
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
OUTPUT_DIR = "models/misinfo_classifier"
MAX_LEN = 128  # Reduced for speed (was 192)
NUM_LABELS = 2  # fake, real
METRIC = "f1"
OPTUNA_TRIALS = 3  # Fast training
SEED = 42
DATA_DIR = "../../data"
MAX_PER_DATASET = 8000  # Reduced for faster training (was 15000)
# =======================================================

LABEL_MAP = {"fake": 0, "real": 1}
ID2LABEL = {0: "fake", 1: "real"}


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
    if pd.isna(text):
        return ""
    text = str(text).lower().strip()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()[:1500]


def load_liar_dataset():
    """Load LIAR dataset (PolitiFact fact-checks)."""
    print("\n[INFO] Loading LIAR dataset...")
    
    try:
        # LIAR has 6 labels - convert to binary
        # pants-fire, false, barely-true -> fake (0)
        # half-true, mostly-true, true -> real (1)
        train = pd.read_csv(os.path.join(DATA_DIR, "liar", "train.tsv"), sep="\t", header=None)
        valid = pd.read_csv(os.path.join(DATA_DIR, "liar", "valid.tsv"), sep="\t", header=None)
        test = pd.read_csv(os.path.join(DATA_DIR, "liar", "test.tsv"), sep="\t", header=None)
        
        df = pd.concat([train, valid, test], ignore_index=True)
        
        # Column 1 = label, Column 2 = statement
        df["text"] = df[2].apply(normalize)
        
        fake_labels = ["pants-fire", "false", "barely-true"]
        df["label"] = df[1].apply(lambda x: 0 if x in fake_labels else 1)
        
        print(f"  ✓ LIAR: {len(df)} samples")
        print(f"    Fake: {(df['label']==0).sum()}, Real: {(df['label']==1).sum()}")
        
        return df[["text", "label"]]
        
    except Exception as e:
        print(f"  ✗ LIAR failed: {e}")
        return pd.DataFrame(columns=["text", "label"])


def load_coaid_dataset():
    """Load CoAID COVID misinformation dataset."""
    print("\n[INFO] Loading CoAID dataset...")
    
    try:
        df = pd.read_csv(os.path.join(DATA_DIR, "coaid", "coaid_claims.csv"))
        
        df["text"] = df["claim"].apply(normalize) if "claim" in df.columns else df.iloc[:, 0].apply(normalize)
        
        # Label column varies - check for common names
        if "label" in df.columns:
            df["label"] = df["label"].apply(lambda x: 0 if str(x).lower() in ["fake", "false", "0"] else 1)
        elif "veracity" in df.columns:
            df["label"] = df["veracity"].apply(lambda x: 0 if str(x).lower() in ["fake", "false", "0"] else 1)
        else:
            df["label"] = 0  # Default to fake for COVID misinfo claims
        
        print(f"  ✓ CoAID: {len(df)} samples")
        return df[["text", "label"]]
        
    except Exception as e:
        print(f"  ✗ CoAID failed: {e}")
        return pd.DataFrame(columns=["text", "label"])


def load_isot_from_huggingface():
    """Load ISOT Fake News dataset from HuggingFace."""
    print("\n[INFO] Loading ISOT/FakeNews dataset from HuggingFace...")
    
    try:
        from datasets import load_dataset
        
        # GonzaloA/fake_news is a well-known fake news dataset
        ds = load_dataset("GonzaloA/fake_news", split="train")
        ds = ds.shuffle(seed=SEED).select(range(min(MAX_PER_DATASET, len(ds))))
        
        df = ds.to_pandas()
        df["text"] = df["text"].apply(normalize)
        df["label"] = df["label"].astype(int)  # 0=fake, 1=real
        
        print(f"  ✓ ISOT/FakeNews: {len(df)} samples")
        return df[["text", "label"]]
        
    except Exception as e:
        print(f"  ✗ ISOT failed: {e}")
        return pd.DataFrame(columns=["text", "label"])


def load_politifact_dataset():
    """Load PolitiFact data if available."""
    print("\n[INFO] Loading PolitiFact dataset...")
    
    try:
        path = os.path.join(DATA_DIR, "politifact")
        files = os.listdir(path)
        
        dfs = []
        for f in files:
            if f.endswith(".csv"):
                temp = pd.read_csv(os.path.join(path, f))
                dfs.append(temp)
        
        if not dfs:
            raise FileNotFoundError("No CSV files found")
            
        df = pd.concat(dfs, ignore_index=True)
        
        # Find text column
        text_col = None
        for col in ["text", "statement", "claim", "content"]:
            if col in df.columns:
                text_col = col
                break
        
        if text_col is None:
            text_col = df.columns[0]
        
        df["text"] = df[text_col].apply(normalize)
        
        # Find label column
        if "label" in df.columns:
            fake_labels = ["pants-fire", "false", "barely-true", "fake", "0"]
            df["label"] = df["label"].apply(lambda x: 0 if str(x).lower() in fake_labels else 1)
        else:
            df["label"] = 0
        
        print(f"  ✓ PolitiFact: {len(df)} samples")
        return df[["text", "label"]]
        
    except Exception as e:
        print(f"  ✗ PolitiFact failed: {e}")
        return pd.DataFrame(columns=["text", "label"])


class MisinfoDataset(torch.utils.data.Dataset):
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
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    report = classification_report(labels, preds, output_dict=True, zero_division=0)
    return {
        "accuracy": report["accuracy"],
        "f1": report["weighted avg"]["f1-score"],
    }


def run_optuna_search(train_ds, val_ds, class_weights, n_trials=OPTUNA_TRIALS):
    def objective(trial):
        lr = trial.suggest_float("lr", 1e-5, 3e-5, log=True)
        batch_size = 16  # Fixed for 4GB VRAM
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
            gradient_accumulation_steps=4,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="steps",
            logging_steps=50,
            load_best_model_at_end=True,
            metric_for_best_model=METRIC,
            report_to="none",
            fp16=True,
            dataloader_num_workers=0,
            dataloader_pin_memory=True,
            optim="adamw_torch_fused",
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
    
    print(f"\n[INFO] Best F1: {study.best_value:.4f}")
    print(f"[INFO] Best params: {study.best_params}")
    
    return study.best_params


def train_final_model(train_ds, val_ds, test_ds, class_weights, best_params, test_df):
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
        num_train_epochs=3,
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

    print("\n[INFO] Evaluating on test set...")
    predictions = trainer.predict(test_ds)
    y_true = predictions.label_ids
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_probs = torch.softmax(torch.tensor(predictions.predictions), dim=-1).numpy()

    print("\n" + "=" * 50)
    print("TEST RESULTS")
    print("=" * 50)
    print(classification_report(y_true, y_pred, target_names=["Fake", "Real"]))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    # Save predictions with test_df
    test_df = test_df.copy()
    test_df["predicted"] = y_pred
    test_df["predicted_label"] = test_df["predicted"].map(ID2LABEL)
    test_df["confidence"] = y_probs.max(axis=1)
    test_df["correct"] = test_df["label"] == test_df["predicted"]
    test_df.to_csv(os.path.join(OUTPUT_DIR, "test_predictions.csv"), index=False)
    print(f"[INFO] Test predictions saved to: {OUTPUT_DIR}/test_predictions.csv")

    trainer.save_model(OUTPUT_DIR)
    print(f"[INFO] Model saved to: {OUTPUT_DIR}")

    return trainer, model


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = get_device()

    # Load all datasets
    dfs = []
    
    liar_df = load_liar_dataset()
    if len(liar_df) > 0:
        dfs.append(liar_df)
    
    coaid_df = load_coaid_dataset()
    if len(coaid_df) > 0:
        dfs.append(coaid_df)
    
    isot_df = load_isot_from_huggingface()
    if len(isot_df) > 0:
        dfs.append(isot_df)
    
    politifact_df = load_politifact_dataset()
    if len(politifact_df) > 0:
        dfs.append(politifact_df)

    if not dfs:
        raise RuntimeError("No datasets loaded!")

    # Combine
    df = pd.concat(dfs, ignore_index=True)
    df = df.dropna()
    df = df[df["text"].str.len() > 10]  # Remove very short texts
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    
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
    train_ds = MisinfoDataset(train_df["text"], train_df["label"], tokenizer)
    val_ds = MisinfoDataset(val_df["text"], val_df["label"], tokenizer)
    test_ds = MisinfoDataset(test_df["text"], test_df["label"], tokenizer)

    # Class weights
    classes = np.unique(train_df["label"])
    weights = compute_class_weight("balanced", classes=classes, y=train_df["label"])
    class_weights = torch.tensor(weights, dtype=torch.float)
    print(f"[INFO] Class weights: {dict(zip(classes.tolist(), weights.tolist()))}")

    # Optuna
    best_params = run_optuna_search(train_ds, val_ds, class_weights)

    # Final training
    train_final_model(train_ds, val_ds, test_ds, class_weights, best_params, test_df)

    print("\n[INFO] Misinformation model training complete!")


if __name__ == "__main__":
    main()
