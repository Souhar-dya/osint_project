#!/usr/bin/env python3
"""
Framing/Propaganda Detection Model Training
- Model: DeBERTa-v3-base
- Datasets: SemEval Propaganda, Media Frames
- Classes: Multiple propaganda techniques + framing categories
- Fast Optuna tuning (3 trials)
"""

import os
import re
import json
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
OUTPUT_DIR = "models/framing_classifier"
MAX_LEN = 256
METRIC = "f1"
OPTUNA_TRIALS = 3
SEED = 42
DATA_DIR = "../../data"
# =======================================================

# Framing categories
FRAME_LABELS = [
    "economic",
    "political", 
    "health",
    "security",
    "environmental",
    "social",
    "legal",
    "other"
]
NUM_LABELS = len(FRAME_LABELS)
LABEL_MAP = {label: i for i, label in enumerate(FRAME_LABELS)}
ID2LABEL = {i: label for i, label in enumerate(FRAME_LABELS)}


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


def load_framing_enhanced():
    """Load enhanced framing dataset."""
    print("\n[INFO] Loading Framing Enhanced dataset...")
    
    try:
        path = os.path.join(DATA_DIR, "framing_enhanced", "train.csv")
        if not os.path.exists(path):
            path = os.path.join(DATA_DIR, "framing_enhanced", "train_large.csv")
        
        df = pd.read_csv(path)
        
        # Find text and label columns
        text_col = "text" if "text" in df.columns else df.columns[0]
        label_col = "label" if "label" in df.columns else "frame" if "frame" in df.columns else df.columns[1]
        
        df["text"] = df[text_col].apply(normalize)
        df["frame"] = df[label_col].str.lower().str.strip()
        
        # Map to our labels
        df["label"] = df["frame"].apply(lambda x: LABEL_MAP.get(x, LABEL_MAP["other"]))
        
        print(f"  ✓ Framing Enhanced: {len(df)} samples")
        print(f"    Frames: {df['frame'].value_counts().to_dict()}")
        
        return df[["text", "label"]]
        
    except Exception as e:
        print(f"  ✗ Framing Enhanced failed: {e}")
        return pd.DataFrame(columns=["text", "label"])


def load_media_frames():
    """Load Media Frames corpus."""
    print("\n[INFO] Loading Media Frames dataset...")
    
    try:
        path = os.path.join(DATA_DIR, "media_frames", "frames.json")
        
        with open(path, "r") as f:
            data = json.load(f)
        
        texts = []
        labels = []
        
        for item in data:
            text = item.get("text", item.get("sentence", ""))
            frame = item.get("frame", item.get("label", "other")).lower()
            
            if text:
                texts.append(normalize(text))
                labels.append(LABEL_MAP.get(frame, LABEL_MAP["other"]))
        
        df = pd.DataFrame({"text": texts, "label": labels})
        print(f"  ✓ Media Frames: {len(df)} samples")
        
        return df
        
    except Exception as e:
        print(f"  ✗ Media Frames failed: {e}")
        return pd.DataFrame(columns=["text", "label"])


def load_ag_news_for_framing():
    """Use AG News as framing proxy (topic classification)."""
    print("\n[INFO] Loading AG News for framing...")
    
    try:
        from datasets import load_dataset
        
        ds = load_dataset("ag_news", split="train")
        ds = ds.shuffle(seed=SEED).select(range(min(20000, len(ds))))
        
        df = ds.to_pandas()
        df["text"] = df["text"].apply(normalize)
        
        # AG News labels: 0=World, 1=Sports, 2=Business, 3=Sci/Tech
        # Map to our frames
        ag_to_frame = {
            0: LABEL_MAP["political"],  # World -> Political
            1: LABEL_MAP["social"],     # Sports -> Social
            2: LABEL_MAP["economic"],   # Business -> Economic
            3: LABEL_MAP["other"],      # Sci/Tech -> Other
        }
        df["label"] = df["label"].map(ag_to_frame)
        
        print(f"  ✓ AG News: {len(df)} samples")
        return df[["text", "label"]]
        
    except Exception as e:
        print(f"  ✗ AG News failed: {e}")
        return pd.DataFrame(columns=["text", "label"])


def generate_synthetic_framing_data():
    """Generate synthetic training data for framing."""
    print("\n[INFO] Generating synthetic framing data...")
    
    templates = {
        "economic": [
            "The stock market {verb} as investors {action}.",
            "GDP growth {verb} to {number}% this quarter.",
            "Unemployment rates {verb} due to {cause}.",
            "The budget deficit {verb} significantly.",
            "Trade tariffs {verb} impacting businesses.",
            "Inflation concerns {verb} consumer spending.",
        ],
        "political": [
            "The government {verb} new legislation on {topic}.",
            "Politicians {verb} over the proposed bill.",
            "Elections {verb} with candidates {action}.",
            "The administration {verb} policy changes.",
            "Congress {verb} to address the issue.",
            "Diplomatic relations {verb} between nations.",
        ],
        "health": [
            "Health officials {verb} about the disease outbreak.",
            "Vaccination rates {verb} across the country.",
            "Hospitals {verb} with patient capacity.",
            "Mental health awareness {verb} importance.",
            "New medical research {verb} promising results.",
            "Healthcare costs {verb} affecting families.",
        ],
        "security": [
            "Security forces {verb} against threats.",
            "Cybersecurity experts {verb} new vulnerabilities.",
            "Border security {verb} priority for officials.",
            "Crime rates {verb} in urban areas.",
            "Military operations {verb} in the region.",
            "Terrorism concerns {verb} heightened vigilance.",
        ],
        "environmental": [
            "Climate change {verb} environmental policies.",
            "Carbon emissions {verb} global targets.",
            "Deforestation {verb} wildlife habitats.",
            "Renewable energy {verb} adoption rates.",
            "Pollution levels {verb} public health.",
            "Conservation efforts {verb} protecting species.",
        ],
        "social": [
            "Community organizations {verb} support programs.",
            "Social media {verb} public discourse.",
            "Education reforms {verb} student outcomes.",
            "Cultural events {verb} celebrating diversity.",
            "Housing affordability {verb} urban residents.",
            "Social inequality {verb} policy debates.",
        ],
        "legal": [
            "The court {verb} landmark decision.",
            "Legal experts {verb} the ruling implications.",
            "Constitutional rights {verb} protection.",
            "Lawsuits {verb} corporate practices.",
            "Regulatory agencies {verb} enforcement actions.",
            "Criminal justice reforms {verb} implementation.",
        ],
    }
    
    verbs = ["shows", "indicates", "reveals", "demonstrates", "suggests", "highlights"]
    actions = ["react to changes", "respond to developments", "face challenges", "see improvements"]
    topics = ["healthcare", "education", "economy", "immigration", "defense", "environment"]
    causes = ["policy changes", "market conditions", "global events", "technological shifts"]
    numbers = ["2.5", "3.1", "4.2", "1.8", "5.0", "2.9"]
    
    data = []
    
    for frame, templates_list in templates.items():
        for template in templates_list:
            for _ in range(50):  # Generate variations
                text = template.format(
                    verb=np.random.choice(verbs),
                    action=np.random.choice(actions),
                    topic=np.random.choice(topics),
                    cause=np.random.choice(causes),
                    number=np.random.choice(numbers),
                )
                data.append({"text": text, "label": LABEL_MAP[frame]})
    
    df = pd.DataFrame(data)
    print(f"  ✓ Synthetic: {len(df)} samples")
    
    return df


class FramingDataset(torch.utils.data.Dataset):
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
    print(classification_report(y_true, y_pred, target_names=FRAME_LABELS))
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
    
    framing_df = load_framing_enhanced()
    if len(framing_df) > 0:
        dfs.append(framing_df)
    
    media_df = load_media_frames()
    if len(media_df) > 0:
        dfs.append(media_df)
    
    ag_df = load_ag_news_for_framing()
    if len(ag_df) > 0:
        dfs.append(ag_df)
    
    # Add synthetic data
    synthetic_df = generate_synthetic_framing_data()
    dfs.append(synthetic_df)

    # Combine
    df = pd.concat(dfs, ignore_index=True)
    df = df.dropna()
    df = df[df["text"].str.len() > 10]
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
    train_ds = FramingDataset(train_df["text"], train_df["label"], tokenizer)
    val_ds = FramingDataset(val_df["text"], val_df["label"], tokenizer)
    test_ds = FramingDataset(test_df["text"], test_df["label"], tokenizer)

    # Class weights
    classes = np.unique(train_df["label"])
    weights = compute_class_weight("balanced", classes=classes, y=train_df["label"])
    class_weights = torch.tensor(weights, dtype=torch.float)

    # Optuna
    best_params = run_optuna_search(train_ds, val_ds, class_weights)

    # Final training
    train_final_model(train_ds, val_ds, test_ds, class_weights, best_params, test_df)

    print("\n[INFO] Framing model training complete!")


if __name__ == "__main__":
    main()
