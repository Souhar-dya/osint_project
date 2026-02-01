"""
Dataset Loaders for Fine-tuning
Supports: FakeNewsNet, LIAR, COVID-19 Fake News, Media Frames
"""
import os
import json
import csv
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class MisinfoSample:
    """Single misinformation dataset sample"""
    text: str
    label: int  # 0 = real, 1 = fake
    source: str
    metadata: Optional[Dict] = None


@dataclass
class FramingSample:
    """Single framing dataset sample"""
    text: str
    frames: List[str]  # Multi-label
    propaganda_techniques: List[str]
    source: str


# =============================================================================
# MISINFORMATION DATASETS
# =============================================================================

class FakeNewsNetLoader:
    """
    FakeNewsNet Dataset Loader
    Contains news from PolitiFact and GossipCop
    
    Structure:
    - politifact_fake/
    - politifact_real/
    - gossipcop_fake/
    - gossipcop_real/
    """
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        
    def load(self, subset: str = "politifact") -> List[MisinfoSample]:
        """Load FakeNewsNet data"""
        samples = []
        
        for label_name, label_id in [("real", 0), ("fake", 1)]:
            folder = self.data_dir / f"{subset}_{label_name}"
            
            if not folder.exists():
                logger.warning(f"Folder not found: {folder}")
                continue
            
            for news_dir in folder.iterdir():
                if news_dir.is_dir():
                    news_file = news_dir / "news content.json"
                    if news_file.exists():
                        try:
                            with open(news_file, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                            
                            text = data.get('text', '') or data.get('title', '')
                            if text:
                                samples.append(MisinfoSample(
                                    text=text[:2000],  # Truncate
                                    label=label_id,
                                    source=f"fakenewsnet_{subset}",
                                    metadata={"title": data.get('title', '')}
                                ))
                        except Exception as e:
                            logger.debug(f"Error loading {news_file}: {e}")
        
        logger.info(f"Loaded {len(samples)} samples from FakeNewsNet ({subset})")
        return samples


class LIARLoader:
    """
    LIAR Dataset Loader
    6-class: pants-fire, false, barely-true, half-true, mostly-true, true
    
    TSV format: id, label, statement, subject, speaker, job, state, party, 
                barely_true_counts, false_counts, half_true_counts, 
                mostly_true_counts, pants_on_fire_counts, context
    """
    
    LABEL_MAP = {
        "pants-fire": 1,  # fake
        "false": 1,
        "barely-true": 1,
        "half-true": 0,   # real (borderline)
        "mostly-true": 0,
        "true": 0
    }
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
    
    def load(self, split: str = "train") -> List[MisinfoSample]:
        """Load LIAR data (train/valid/test)"""
        samples = []
        
        file_path = self.data_dir / f"{split}.tsv"
        if not file_path.exists():
            logger.warning(f"LIAR file not found: {file_path}")
            return self._get_sample_data()
        
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if len(row) >= 3:
                    label_str = row[1].lower()
                    statement = row[2]
                    
                    if label_str in self.LABEL_MAP:
                        samples.append(MisinfoSample(
                            text=statement,
                            label=self.LABEL_MAP[label_str],
                            source="liar",
                            metadata={"original_label": label_str}
                        ))
        
        logger.info(f"Loaded {len(samples)} samples from LIAR ({split})")
        return samples
    
    def _get_sample_data(self) -> List[MisinfoSample]:
        """Return sample data for testing"""
        return [
            MisinfoSample("The economy grew by 5% last quarter according to official reports.", 0, "liar_sample"),
            MisinfoSample("Exposed: Secret government program to control weather!", 1, "liar_sample"),
            MisinfoSample("Scientists confirm new treatment reduces symptoms by 40%.", 0, "liar_sample"),
            MisinfoSample("BREAKING: Vaccines contain microchips for tracking citizens!", 1, "liar_sample"),
            MisinfoSample("The unemployment rate dropped to 3.5% this month.", 0, "liar_sample"),
            MisinfoSample("They don't want you to know: 5G towers spread the virus!", 1, "liar_sample"),
        ]


class CovidFakeNewsLoader:
    """
    COVID-19 Fake News Dataset Loader
    Binary classification: real/fake COVID-19 news
    """
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
    
    def load(self) -> List[MisinfoSample]:
        """Load COVID fake news data"""
        samples = []
        
        # Try CSV format
        for filename in ["Constraint_Train.csv", "train.csv", "covid_fake_news.csv"]:
            file_path = self.data_dir / filename
            if file_path.exists():
                df = pd.read_csv(file_path)
                
                # Common column names
                text_col = None
                label_col = None
                
                for col in ['tweet', 'text', 'content', 'statement']:
                    if col in df.columns:
                        text_col = col
                        break
                
                for col in ['label', 'class', 'fake', 'is_fake']:
                    if col in df.columns:
                        label_col = col
                        break
                
                if text_col and label_col:
                    for _, row in df.iterrows():
                        text = str(row[text_col])
                        label = row[label_col]
                        
                        # Normalize label
                        if isinstance(label, str):
                            label = 1 if label.lower() in ['fake', 'false', '1'] else 0
                        else:
                            label = int(label)
                        
                        samples.append(MisinfoSample(
                            text=text[:1500],
                            label=label,
                            source="covid_fake_news"
                        ))
                    break
        
        if not samples:
            samples = self._get_sample_data()
        
        logger.info(f"Loaded {len(samples)} samples from COVID Fake News")
        return samples
    
    def _get_sample_data(self) -> List[MisinfoSample]:
        """Sample COVID misinformation data"""
        return [
            MisinfoSample("WHO recommends wearing masks in crowded indoor spaces.", 0, "covid_sample"),
            MisinfoSample("Drinking bleach cures COVID-19! Share before they delete!", 1, "covid_sample"),
            MisinfoSample("Pfizer vaccine shows 95% efficacy in clinical trials.", 0, "covid_sample"),
            MisinfoSample("COVID vaccines alter your DNA and make you magnetic!", 1, "covid_sample"),
            MisinfoSample("Social distancing helps reduce virus transmission rates.", 0, "covid_sample"),
            MisinfoSample("5G towers activated the virus! Wake up sheeple!", 1, "covid_sample"),
            MisinfoSample("Hospitals report decrease in COVID cases after vaccination campaign.", 0, "covid_sample"),
            MisinfoSample("EXPOSED: COVID is a hoax created by big pharma for profits!", 1, "covid_sample"),
        ]


# =============================================================================
# FRAMING & PROPAGANDA DATASETS
# =============================================================================

class MediaFramesLoader:
    """
    Media Frames Corpus Loader
    Multi-label frame classification
    """
    
    FRAMES = [
        "Economic", "Capacity", "Morality", "Fairness", "Legality",
        "Policy", "Crime", "Security", "Health", "Quality_of_Life",
        "Cultural", "Public_Opinion", "Political", "External"
    ]
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
    
    def load(self) -> List[FramingSample]:
        """Load Media Frames data"""
        samples = []
        
        # Try CSV format first (our downloaded data)
        csv_path = self.data_dir / "media_frames.csv"
        if csv_path.exists():
            try:
                import pandas as pd
                df = pd.read_csv(csv_path)
                for _, row in df.iterrows():
                    frame = row.get('frame', row.get('frames', ''))
                    frames = [frame] if isinstance(frame, str) else frame
                    samples.append(FramingSample(
                        text=str(row.get('text', '')),
                        frames=frames if isinstance(frames, list) else [frames],
                        propaganda_techniques=[],
                        source="media_frames"
                    ))
            except Exception as e:
                logger.warning(f"Error loading CSV: {e}")
        
        # Try JSON formats
        if not samples:
            for filename in ["media_frames.json", "train.json"]:
                file_path = self.data_dir / filename
                if file_path.exists():
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    for item in data:
                        samples.append(FramingSample(
                            text=item.get('text', ''),
                            frames=item.get('frames', []),
                            propaganda_techniques=[],
                            source="media_frames"
                        ))
                    break
        
        if not samples:
            samples = self._get_sample_data()
        
        logger.info(f"Loaded {len(samples)} samples from Media Frames")
        return samples
    
    def _get_sample_data(self) -> List[FramingSample]:
        """Sample framing data"""
        return [
            FramingSample(
                "The new tax bill will cost families an average of $2000 more per year.",
                ["Economic"], [], "media_frames_sample"
            ),
            FramingSample(
                "Critics argue the policy violates constitutional protections.",
                ["Legality", "Political"], [], "media_frames_sample"
            ),
            FramingSample(
                "Health officials warn of potential outbreak if vaccines rates decline.",
                ["Health", "Policy"], [], "media_frames_sample"
            ),
            FramingSample(
                "The military budget increase is necessary for national defense.",
                ["Security", "Economic"], [], "media_frames_sample"
            ),
            FramingSample(
                "Protesters demand equal treatment under the law.",
                ["Fairness", "Legality"], [], "media_frames_sample"
            ),
        ]


class SemEvalPropagandaLoader:
    """
    SemEval-2020 Task 11: Propaganda Techniques Loader
    Span-level propaganda detection
    """
    
    TECHNIQUES = [
        "Loaded_Language", "Name_Calling", "Repetition", "Exaggeration",
        "Doubt", "Appeal_to_Fear", "Flag_Waving", "Causal_Oversimplification",
        "Slogans", "Appeal_to_Authority", "Black_White_Fallacy",
        "Thought_Terminating", "Whataboutism", "Reductio_ad_Hitlerum",
        "Straw_Man", "Bandwagon", "Obfuscation"
    ]
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
    
    def load(self) -> List[FramingSample]:
        """Load SemEval propaganda data"""
        samples = []
        
        # Try CSV format first (our downloaded data)
        csv_path = self.data_dir / "propaganda.csv"
        if csv_path.exists():
            try:
                import pandas as pd
                df = pd.read_csv(csv_path)
                for _, row in df.iterrows():
                    technique = row.get('technique', row.get('propaganda_technique', ''))
                    techniques = [technique] if isinstance(technique, str) else technique
                    samples.append(FramingSample(
                        text=str(row.get('text', '')),
                        frames=[],
                        propaganda_techniques=techniques if isinstance(techniques, list) else [techniques],
                        source="propaganda"
                    ))
            except Exception as e:
                logger.warning(f"Error loading propaganda CSV: {e}")
        
        # Try to load from articles directory (original SemEval format)
        if not samples:
            articles_dir = self.data_dir / "train-articles"
            labels_dir = self.data_dir / "train-labels-task-flc-tc"
            
            if articles_dir.exists() and labels_dir.exists():
                for article_file in articles_dir.glob("article*.txt"):
                    article_id = article_file.stem
                    label_file = labels_dir / f"{article_id}.task-flc-tc.labels"
                    
                    with open(article_file, 'r', encoding='utf-8') as f:
                        text = f.read()
                    
                    techniques = []
                    if label_file.exists():
                        with open(label_file, 'r') as f:
                            for line in f:
                                parts = line.strip().split('\t')
                                if len(parts) >= 4:
                                    techniques.append(parts[1])
                    
                    samples.append(FramingSample(
                        text=text[:2000],
                        frames=[],
                        propaganda_techniques=list(set(techniques)),
                        source="semeval_propaganda"
                    ))
        
        if not samples:
            samples = self._get_sample_data()
        
        logger.info(f"Loaded {len(samples)} samples from SemEval Propaganda")
        return samples
    
    def _get_sample_data(self) -> List[FramingSample]:
        """Sample propaganda data"""
        return [
            FramingSample(
                "The radical left wants to DESTROY our great nation!",
                [], ["Loaded_Language", "Flag_Waving", "Exaggeration"], "semeval_sample"
            ),
            FramingSample(
                "Everyone knows this policy is a disaster. You're either with us or against us.",
                [], ["Bandwagon", "Black_White_Fallacy"], "semeval_sample"
            ),
            FramingSample(
                "What about when they did the same thing? Typical hypocrisy!",
                [], ["Whataboutism", "Name_Calling"], "semeval_sample"
            ),
            FramingSample(
                "Experts agree this is the only solution. Don't question it.",
                [], ["Appeal_to_Authority", "Thought_Terminating"], "semeval_sample"
            ),
        ]


# =============================================================================
# UNIFIED DATA LOADER
# =============================================================================

class UnifiedDataLoader:
    """Load and combine multiple datasets"""
    
    def __init__(self, data_root: str = "./data"):
        self.data_root = Path(data_root)
    
    def load_misinfo_data(self) -> Tuple[List[MisinfoSample], List[MisinfoSample]]:
        """Load all misinformation datasets, return train/test split"""
        all_samples = []
        
        # FakeNewsNet
        fnn_loader = FakeNewsNetLoader(self.data_root / "fakenewsnet")
        all_samples.extend(fnn_loader.load("politifact"))
        
        # LIAR
        liar_loader = LIARLoader(self.data_root / "liar")
        all_samples.extend(liar_loader.load("train"))
        
        # COVID
        covid_loader = CovidFakeNewsLoader(self.data_root / "covid_fake_news")
        all_samples.extend(covid_loader.load())
        
        # If no data loaded, use samples
        if not all_samples:
            logger.warning("No data files found, using sample data")
            all_samples = liar_loader._get_sample_data()
            all_samples.extend(covid_loader._get_sample_data())
        
        # Shuffle and split
        import random
        random.shuffle(all_samples)
        split_idx = int(len(all_samples) * 0.8)
        
        return all_samples[:split_idx], all_samples[split_idx:]
    
    def load_framing_data(self) -> Tuple[List[FramingSample], List[FramingSample]]:
        """Load all framing datasets, return train/test split"""
        all_samples = []
        
        # Media Frames
        mf_loader = MediaFramesLoader(self.data_root / "media_frames")
        all_samples.extend(mf_loader.load())
        
        # Propaganda (from propaganda folder)
        propaganda_loader = SemEvalPropagandaLoader(self.data_root / "propaganda")
        all_samples.extend(propaganda_loader.load())
        
        # Also check semeval_propaganda folder for backward compatibility
        if (self.data_root / "semeval_propaganda").exists():
            semeval_loader = SemEvalPropagandaLoader(self.data_root / "semeval_propaganda")
            all_samples.extend(semeval_loader.load())
        
        # Shuffle and split
        import random
        random.shuffle(all_samples)
        split_idx = int(len(all_samples) * 0.8)
        
        return all_samples[:split_idx], all_samples[split_idx:]


if __name__ == "__main__":
    # Test loaders
    loader = UnifiedDataLoader("./data")
    
    train, test = loader.load_misinfo_data()
    print(f"Misinfo - Train: {len(train)}, Test: {len(test)}")
    
    train_f, test_f = loader.load_framing_data()
    print(f"Framing - Train: {len(train_f)}, Test: {len(test_f)}")
