# AI-Based Social Media & OSINT Monitoring Chrome Extension

A real-time browser-embedded OSINT system that analyzes public social media content for sentiment, narrative framing, misinformation, and propaganda detection.

## 🎯 Research Novelty

**Narrative Distortion Measurement**: Unlike existing tools, this system quantifies how far social media narratives deviate from verified event baselines (GDELT, FEVER), providing explainable intelligence outputs.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Chrome Extension                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │Content Script│  │  Popup UI   │  │ Background  │         │
│  │(DOM Extract) │  │  (React)    │  │  (Service)  │         │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │
└─────────┼────────────────┼────────────────┼─────────────────┘
          │                │                │
          └────────────────┴────────────────┘
                           │
                    ┌──────▼──────┐
                    │  FastAPI    │
                    │  Backend    │
                    └──────┬──────┘
                           │
     ┌─────────────────────┼─────────────────────┐
     │                     │                     │
┌────▼────┐  ┌─────────────▼───────────┐  ┌─────▼─────┐
│Sentiment│  │  Narrative & Framing    │  │ Misinfo   │
│ Module  │  │       Module            │  │  Module   │
└────┬────┘  └─────────────┬───────────┘  └─────┬─────┘
     │                     │                     │
     └─────────────────────┼─────────────────────┘
                           │
                    ┌──────▼──────┐
                    │ Baseline    │◄──── GDELT / FEVER
                    │ Comparison  │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │Explainability│
                    │   Layer     │
                    └─────────────┘
```

## 📁 Project Structure

```
ProjectSem8/
├── backend/                 # FastAPI Backend
│   ├── app/
│   │   ├── main.py         # Entry point
│   │   ├── config.py       # Configuration
│   │   ├── database.py     # SQLAlchemy models
│   │   ├── models/         # Pydantic schemas
│   │   ├── routers/        # API endpoints
│   │   └── services/       # AI inference modules
│   │       ├── sentiment.py
│   │       ├── topics.py
│   │       ├── framing.py
│   │       ├── misinfo.py
│   │       ├── baseline.py
│   │       └── explainer.py
│   ├── requirements.txt
│   └── Dockerfile
├── extension/              # Chrome Extension (Manifest V3)
│   ├── manifest.json
│   ├── background.js
│   ├── content.js
│   ├── popup/
│   └── styles/
├── ml/                     # ML Modules
│   ├── baseline/           # GDELT/FEVER loaders
│   └── framing/            # Frame classifier
├── evaluation/             # Metrics & case studies
├── database/               # SQL schemas
├── docker-compose.yml
└── README.md
```

## 🚀 Quick Start

### Option 1: Local Development

```bash
# 1. Clone and navigate
cd ProjectSem8

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 3. Install dependencies
cd backend
pip install -r requirements.txt

# 4. Run the backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 5. Load the Chrome extension
# - Open chrome://extensions/
# - Enable "Developer mode"
# - Click "Load unpacked"
# - Select the `extension` folder
```

### Option 2: Docker

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f backend

# Stop services
docker-compose down
```

## 📡 API Endpoints

### Analyze Text
```http
POST /api/analyze
Content-Type: application/json

{
  "text": "Breaking news about climate change...",
  "source": "twitter",
  "anonymize": true,
  "include_baseline": true
}
```

### Response
```json
{
  "sentiment": {
    "label": "negative",
    "score": 0.85,
    "emotions": {"fear": 0.6}
  },
  "topics": {
    "topic_id": 4,
    "topic_label": "Climate/Environment",
    "keywords": ["climate", "change", "warming"]
  },
  "framing": {
    "frame": "Security",
    "propaganda_techniques": ["fear_appeal"],
    "confidence": 0.72
  },
  "misinformation": {
    "risk_score": 0.45,
    "risk_level": "medium",
    "triggers": ["sensational language"]
  },
  "baseline": {
    "narrative_distance": 0.35,
    "closest_event": "IPCC Climate Report 2024",
    "deviation_type": "exaggeration"
  },
  "explanation": {
    "confidence": 0.78,
    "key_phrases": ["breaking", "climate"],
    "reasoning": "Content shows moderate deviation from verified climate data...",
    "flags": ["⚠️ MODERATE_DISTORTION"]
  }
}
```

## 🧠 AI Modules

| Module | Model | Purpose |
|--------|-------|---------|
| Sentiment | RoBERTa (Twitter) | Emotion & polarity detection |
| Topics | SBERT + BERTopic | Narrative clustering |
| Framing | Zero-shot BART | Media frame classification |
| Misinfo | Rule-based + ML | Misinformation indicators |
| Baseline | SBERT cosine | Event-grounded comparison |

## 📊 Evaluation

Run evaluation scripts:
```bash
cd evaluation
python metrics.py
```

See [case_studies.md](evaluation/case_studies.md) for detailed examples.

## ⚖️ Ethical Compliance

- ✅ Analyzes only **publicly visible** content
- ✅ **Anonymizes** @mentions and usernames
- ✅ **Does not store** raw text (only hashes)
- ✅ User-controlled settings for privacy
- ✅ Transparent explainability outputs

## 🔧 Configuration

Edit `backend/.env`:
```env
DATABASE_URL=sqlite:///./osint_logs.db
SENTIMENT_MODEL=cardiffnlp/twitter-roberta-base-sentiment-latest
ANONYMIZE_TEXT=true
LOG_ANALYSIS=true
```

## 📚 Datasets Used

- **Sentiment**: TweetEval, GoEmotions, Sentiment140
- **Framing**: Media Frames Corpus, SemEval Propaganda
- **Misinfo**: FakeNewsNet, LIAR, COVID-19 Fake News
- **Baseline**: FEVER (claims), GDELT (events)

## 📝 License

MIT License - See LICENSE file

## 👥 Authors

Souhardya Kundu
Debanshu Prusty
Subham Agarwal


## 🙏 Acknowledgments

- HuggingFace Transformers
- GDELT Project
- FEVER Dataset Authors
