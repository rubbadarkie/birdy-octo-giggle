# Bird Species Recognition and Educational Assistant for Singapore

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A two-component system for Singapore bird species identification and ecological education, developed as a Final Year Project at Nanyang Technological University (NTU).

## Overview

This project addresses two challenges in citizen science bird monitoring:

1. **Acoustic Classification**: Global models like BirdNET underperform on Southeast Asian species due to training data bias. We improve recognition accuracy for Singapore garden birds using transfer learning.

2. **Ecological Education**: General-purpose LLMs lack Singapore-specific knowledge. We provide locally grounded educational content using retrieval-augmented generation (RAG).

### Key Results

| Component | Metric | Result |
|-----------|--------|--------|
| **Classifier** | Accuracy (5-fold CV) | 62.7% ± 6.3% (vs 32.1% baseline) |
| **Classifier** | Relative Improvement | +95.3% |
| **RAG Pipeline** | Evaluation Score | 22.2/25 (vs 14.2/25 baseline LLM) |
| **RAG Pipeline** | Singapore-Specificity | 5.0/5.0 (vs 2.2/5.0 baseline) |

---

## Table of Contents

- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
  - [Classifier](#classifier)
  - [RAG Pipeline](#rag-pipeline)
- [Training](#training)
- [Evaluation](#evaluation)
- [Project Structure](#project-structure)
- [Limitations](#limitations)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Features

### Classifier Component
- Feature-based transfer learning using BirdNET V2.4 as frozen feature extractor
- MLP classifier trained on 506 Singapore/Southeast Asian recordings
- 26 garden bird species (Rock Pigeon excluded due to insufficient data)
- 5-fold stratified cross-validation with matched baseline evaluation
- Data augmentation pipeline (6 transformations per recording)

### RAG Component
- Knowledge base from NParks Garden Bird Watch + Singapore Bird Database
- 50 species with structured metadata (conservation status, residency, vernacular names)
- Hybrid query routing (semantic search + metadata filtering)
- ChromaDB vector storage with all-MiniLM-L6-v2 embeddings
- Claude Sonnet 4 for response generation

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         END-TO-END PIPELINE                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  [Audio Input]                                                           │
│       │                                                                  │
│       ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                   CLASSIFIER COMPONENT                           │    │
│  │                                                                  │    │
│  │  Audio (5s) → BirdNET V2.4 → 6,522-dim → MLP → Species + Conf   │    │
│  │               (frozen)       features    (trained)               │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│       │                                                                  │
│       ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                      RAG COMPONENT                               │    │
│  │                                                                  │    │
│  │  Query → Embedding → ChromaDB → Retrieved Chunks → Claude → Response│
│  │          (MiniLM)    Retrieval   + Metadata        Sonnet 4      │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│       │                                                                  │
│       ▼                                                                  │
│  [Educational Response with Singapore-specific context]                  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Installation

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (optional, for faster training)
- ~2GB disk space for models and data

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/birdnet-sg.git
cd birdnet-sg

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```txt
# requirements.txt

# Core
torch>=2.0.0
torchaudio>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0

# Audio processing
librosa>=0.10.0
soundfile>=0.12.0
audiomentations>=0.30.0

# BirdNET
birdnetlib>=0.15.0

# RAG pipeline
chromadb>=0.4.0
sentence-transformers>=2.2.0
anthropic>=0.18.0
pymupdf>=1.23.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
```

---

## Dataset

### Audio Recordings

| Source | Recordings | Species | Geographic Priority |
|--------|------------|---------|---------------------|
| Xeno-canto | 509 | 27 | Singapore > Malaysia > Indonesia |

**Species Distribution:**
- 22 species with 20 recordings each
- 4 species with 15-19 recordings
- 1 species excluded (Rock Pigeon, n=3)

**Excluded Species (Xeno-canto restrictions):**
- Javan Myna
- Oriental Magpie-Robin  
- Black-naped Oriole

### Knowledge Base

| Source | Coverage | Content |
|--------|----------|---------|
| NParks Garden Bird Watch | 50 species | Descriptions, habitat, behaviour, trends |
| Singapore Bird Database | All species | Status, RDB3, vernacular names |

### Data Availability

Due to licensing restrictions, raw audio files are not included. To reproduce:

1. Download recordings from [Xeno-canto](https://xeno-canto.org/) using species list in `data/species_list.csv`
2. Run preprocessing: `python scripts/preprocess_audio.py`
3. Extract features: `python scripts/extract_features.py`

---

## Usage

### Classifier

#### Quick Inference

```python
from src.classifier import BirdClassifier

# Load trained model
classifier = BirdClassifier.load("models/classifier_final.pth")

# Predict species from audio file
prediction = classifier.predict("path/to/audio.mp3")

print(f"Species: {prediction['species']}")
print(f"Confidence: {prediction['confidence']:.1%}")
```

#### Batch Processing

```python
from src.classifier import BirdClassifier
import glob

classifier = BirdClassifier.load("models/classifier_final.pth")

# Process multiple files
audio_files = glob.glob("recordings/*.mp3")
results = classifier.predict_batch(audio_files)

for result in results:
    print(f"{result['file']}: {result['species']} ({result['confidence']:.1%})")
```

### RAG Pipeline

#### Species Explanation (Post-Classification)

```python
from src.rag import BirdRAG

# Initialize RAG pipeline
rag = BirdRAG(
    chromadb_path="data/chromadb",
    api_key="your-anthropic-api-key"
)

# Get educational explanation for identified species
response = rag.explain_species(
    species="Olive-backed Sunbird",
    confidence=0.87
)

print(response)
```

#### Freeform Queries

```python
# Ask general questions about Singapore birds
response = rag.ask("Which birds in Singapore are migratory?")
print(response)

response = rag.ask("Where can I spot kingfishers?")
print(response)
```

#### Conversational Mode

```python
# Multi-turn conversation
rag.start_conversation()

response1 = rag.ask("Tell me about the Asian Koel")
print(response1)

response2 = rag.ask("Is it native to Singapore?")  # Maintains context
print(response2)

rag.end_conversation()
```

---

## Training

### Classifier Training

```bash
# Full training with 5-fold cross-validation
python scripts/train_classifier.py \
    --data_dir data/processed \
    --output_dir models \
    --n_folds 5 \
    --epochs 200 \
    --patience 25 \
    --batch_size 32 \
    --learning_rate 0.001
```

### RAG Knowledge Base Construction

```bash
# Extract and process documents
python scripts/build_knowledge_base.py \
    --nparks_pdf data/sources/garden_bird_watch.pdf \
    --sg_bird_db data/sources/sg_bird_database.csv \
    --output_dir data/chromadb \
    --chunk_size 200 \
    --chunk_overlap 50
```

---

## Evaluation

### Classifier Evaluation

```bash
# Run 5-fold CV evaluation with matched baseline comparison
python scripts/evaluate_classifier.py \
    --data_dir data/processed \
    --output_dir results/classifier
```

**Output:**
- `cv_fold_results.csv` - Per-fold accuracy
- `cv_species_results.csv` - Per-species accuracy
- `cv_accuracy_comparison.png` - Visualization

### RAG Evaluation (LLM-as-a-Judge)

```bash
# Run automated evaluation
python scripts/evaluate_rag.py \
    --test_queries data/eval/test_queries.json \
    --output_dir results/rag \
    --judge_model claude-opus-4-6-20250116
```

**Output:**
- `llm_judge_results.json` - Full evaluation data
- `llm_judge_summary.csv` - Score summary

---

## Project Structure

```
birdnet-sg/
├── README.md
├── requirements.txt
├── LICENSE
│
├── data/
│   ├── raw/                     # Original audio files (not included)
│   ├── processed/               # Standardised 5-second clips
│   ├── augmented/               # Augmented training data
│   ├── chromadb/                # Vector database
│   ├── sources/                 # PDF and CSV source documents
│   └── species_list.csv         # Target species list
│
├── models/
│   ├── classifier_final.pth     # Trained MLP classifier
│   ├── feature_scaler.pkl       # StandardScaler for features
│   └── species_mapping.json     # Species to index mapping
│
├── src/
│   ├── __init__.py
│   ├── classifier/
│   │   ├── __init__.py
│   │   ├── model.py             # MLP architecture
│   │   ├── features.py          # BirdNET feature extraction
│   │   ├── augmentation.py      # Audio augmentation
│   │   └── inference.py         # Prediction interface
│   │
│   └── rag/
│       ├── __init__.py
│       ├── knowledge_base.py    # Document processing
│       ├── retrieval.py         # ChromaDB retrieval
│       ├── routing.py           # Hybrid query routing
│       └── generation.py        # Claude response generation
│
├── scripts/
│   ├── preprocess_audio.py      # Audio standardisation
│   ├── extract_features.py      # BirdNET feature extraction
│   ├── train_classifier.py      # Training script
│   ├── build_knowledge_base.py  # RAG knowledge base construction
│   ├── evaluate_classifier.py   # Classifier evaluation
│   └── evaluate_rag.py          # RAG evaluation
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_classifier_training.ipynb
│   ├── 03_rag_pipeline.ipynb
│   └── 04_evaluation.ipynb
│
└── results/
    ├── classifier/
    │   ├── cv_fold_results.csv
    │   ├── cv_species_results.csv
    │   └── figures/
    │
    └── rag/
        ├── llm_judge_results.json
        └── response_transcripts/
```

---

## Limitations

### Classifier
- **Limited training data**: ~20 recordings per species; results may not generalise to all acoustic conditions
- **Species exclusions**: 3 species excluded due to Xeno-canto restrictions; 1 excluded due to insufficient data
- **Single-label classification**: Does not handle overlapping vocalisations from multiple species
- **Fixed audio length**: All inputs standardised to 5 seconds; very short vocalisations may be diluted by padding

### RAG Pipeline
- **Knowledge base coverage**: 50 species from NParks; some classifier species lack detailed profiles
- **Location granularity**: Cannot provide neighbourhood-specific recommendations (e.g., "birds in Ang Mo Kio")
- **English only**: Does not support Malay or Chinese queries/responses
- **API dependency**: Requires Anthropic API for response generation

---

## Citation

If you use this work, please cite:

```bibtex
@thesis{ng2026birdnetsg,
  title={BirdNET-SG: Bird Species Recognition and Educational Assistant for Singapore},
  author={Ng, Adele},
  year={2026},
  school={Nanyang Technological University},
  type={Bachelor's Thesis}
}
```

---

## Acknowledgements

- **BirdNET** by [K. Lisa Yang Center for Conservation Bioacoustics](https://birdnet.cornell.edu/) - Pre-trained acoustic model
- **Xeno-canto** - Bird sound recordings database
- **National Parks Board (NParks)** - Garden Bird Watch species profiles
- **Nature Society Singapore** - Singapore Bird Database
- **Anthropic** - Claude API for response generation

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Third-Party Licenses

- Audio recordings from Xeno-canto are subject to individual Creative Commons licenses
- BirdNET is licensed under [Creative Commons Attribution-NonCommercial-ShareAlike 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
- NParks content used for educational/research purposes

---

## Contact

**Adele Ng**  
College of Computing and Data Science
Nanyang Technological University  

For questions or collaboration inquiries, please open an issue or contact via NTU email.
