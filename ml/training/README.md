# Model Fine-tuning Guide

## Quick Start

### 1. Install Dependencies
```bash
cd ml/training
pip install -r requirements.txt
```

### 2. Prepare Data Directory
```
data/
├── fakenewsnet/          # FakeNewsNet dataset
│   ├── politifact_fake/
│   ├── politifact_real/
│   ├── gossipcop_fake/
│   └── gossipcop_real/
├── liar/                 # LIAR dataset
│   ├── train.tsv
│   ├── valid.tsv
│   └── test.tsv
├── covid_fake_news/      # COVID fake news
│   └── train.csv
├── media_frames/         # Media Frames Corpus
│   └── frames.csv
└── semeval_propaganda/   # SemEval 2020 Task 11
    ├── train-articles/
    └── train-labels-task-flc-tc/
```

**Note**: If datasets are not available, sample data will be used automatically.

### 3. Download Datasets

| Dataset | Link | Size |
|---------|------|------|
| FakeNewsNet | [GitHub](https://github.com/KaiDMML/FakeNewsNet) | ~2GB |
| LIAR | [UVic](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip) | ~2MB |
| COVID Fake News | [Kaggle](https://www.kaggle.com/datasets/arashnic/covid19-fake-news) | ~50MB |
| Media Frames | [GitHub](https://github.com/dallascard/media_frames_corpus) | ~100MB |
| SemEval Propaganda | [Propaganda Analysis](https://propaganda.qcri.org/) | ~50MB |

---

## Training Models

### Train Misinformation Classifier
```bash
python train_misinfo.py \
    --data_dir ./data \
    --output_dir ./models/misinfo_classifier \
    --epochs 3 \
    --batch_size 16 \
    --learning_rate 2e-5
```

**Expected Output**:
```
Accuracy:  0.85-0.90
Precision: 0.83-0.88
Recall:    0.82-0.87
F1 Score:  0.83-0.88
```

### Train Framing Classifier (Frames)
```bash
python train_framing.py \
    --task frames \
    --data_dir ./data \
    --output_dir ./models/framing_classifier \
    --epochs 4 \
    --batch_size 16
```

### Train Propaganda Detector
```bash
python train_framing.py \
    --task propaganda \
    --data_dir ./data \
    --output_dir ./models/propaganda_classifier \
    --epochs 4
```

---

## Evaluation

### Evaluate Misinformation Model
```bash
python evaluate.py \
    --model_type misinfo \
    --model_dir ./models/misinfo_classifier \
    --data_dir ./data \
    --output_dir ./evaluation_results/misinfo
```

### Evaluate Framing Model
```bash
python evaluate.py \
    --model_type framing \
    --model_dir ./models/framing_classifier_frames \
    --data_dir ./data \
    --output_dir ./evaluation_results/framing
```

---

## Using Trained Models

### In Python
```python
from train_misinfo import MisinfoClassifier
from train_framing import FramingClassifier

# Misinformation
misinfo = MisinfoClassifier("./models/misinfo_classifier")
result = misinfo.predict("Breaking news about vaccine dangers!")
print(result)
# {'label': 'fake', 'is_fake': True, 'confidence': 0.92, ...}

# Framing
framing = FramingClassifier("./models/framing_classifier_frames")
result = framing.predict("The tax bill will cost families millions.")
print(result)
# {'predicted_labels': ['Economic'], 'top_label': 'Economic', ...}
```

### Integration with Backend
Update `backend/app/services/misinfo.py`:

```python
from ml.training import MisinfoClassifier

# Load once at startup
classifier = MisinfoClassifier("./models/misinfo_classifier")

def detect_misinformation(text: str):
    result = classifier.predict(text)
    return MisinfoResult(
        risk_score=result['probabilities']['fake'],
        risk_level="high" if result['is_fake'] else "low",
        triggers=[]
    )
```

---

## Hyperparameter Tuning

### Recommended Values

| Parameter | Misinformation | Framing |
|-----------|---------------|---------|
| Epochs | 3-5 | 4-6 |
| Batch Size | 16-32 | 16 |
| Learning Rate | 2e-5 | 2e-5 to 3e-5 |
| Max Length | 256 | 256 |
| Warmup Ratio | 0.1 | 0.1 |

### Tips
- Start with fewer epochs, add more if underfitting
- Reduce batch size if GPU memory errors
- Use early stopping to prevent overfitting
- Balance classes for misinformation (real/fake ratio)

---

## Expected Results

### Misinformation Detection
| Metric | Expected Range |
|--------|---------------|
| Accuracy | 0.82 - 0.90 |
| F1 (Fake) | 0.80 - 0.88 |
| ROC-AUC | 0.88 - 0.94 |

### Frame Classification
| Metric | Expected Range |
|--------|---------------|
| Micro F1 | 0.65 - 0.75 |
| Macro F1 | 0.55 - 0.70 |
| Exact Match | 0.40 - 0.55 |

### Propaganda Detection
| Metric | Expected Range |
|--------|---------------|
| Micro F1 | 0.60 - 0.70 |
| Macro F1 | 0.45 - 0.60 |

---

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
python train_misinfo.py --batch_size 8

# Or use CPU (slower)
CUDA_VISIBLE_DEVICES="" python train_misinfo.py
```

### No Data Found
The scripts automatically use sample data if datasets aren't found. For real training, download the datasets.

### Model Not Converging
- Check class balance
- Try lower learning rate (1e-5)
- Increase warmup ratio (0.2)
