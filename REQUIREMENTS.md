# SingBird — Requirements Document
**Singapore Bird Sound Recognition & AI Assistant**
NTU Computer Science FYP | Adele Ng | 2025–2026

---

## Quick Summary

SingBird is a web app where users upload or record bird sounds → the app identifies the Singapore bird species using a fine-tuned BirdNET model → then provides educational information and conversational Q&A powered by a RAG pipeline using Anthropic Claude.

---

## Pre-built Artefacts (DO NOT REGENERATE)

These files already exist locally and must not be modified or regenerated:

| File/Folder | Description |
|---|---|
| `model_files/singapore_bird_classifier_final.pth` | Fine-tuned PyTorch MLP classifier weights |
| `model_files/feature_scaler_final.pkl` | scikit-learn StandardScaler for embedding normalisation |
| `model_files/label_mapping.json` | Maps class indices to species names |
| `chroma_db/` | Pre-built ChromaDB vector store (91 chunks, 50 species) |
| `.env` | API keys — set manually, never hardcode |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 18 + Vite + Tailwind CSS + shadcn/ui |
| Backend | FastAPI (Python 3.10+) |
| ML Model | BirdNET + custom MLP (pre-trained, .pth file) |
| Embeddings | sentence-transformers all-MiniLM-L6-v2 (local) |
| Vector DB | ChromaDB (persistent, pre-built) |
| LLM | Anthropic Claude (claude-sonnet-4-20250514) |
| Audio | librosa + soundfile |
| Frontend hosting | Vercel |
| Backend hosting | Railway |

---

## Project Structure

```
singbird-backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI entry point + CORS
│   ├── config.py            # env vars and file paths
│   ├── models/
│   │   ├── __init__.py
│   │   └── classifier.py    # BirdClassifier class (ML inference)
│   ├── services/
│   │   ├── __init__.py
│   │   ├── audio.py         # audio preprocessing pipeline
│   │   └── rag.py           # RAG pipeline functions
│   └── api/
│       ├── __init__.py
│       └── routes.py        # all API endpoint definitions
├── model_files/             # pre-trained model artefacts
├── chroma_db/               # pre-built ChromaDB vector store
├── requirements.txt
└── .env
```

---

## Environment Variables (.env)

```
ANTHROPIC_API_KEY=your_key_here
MODEL_PATH=./model_files/singapore_bird_classifier_final.pth
SCALER_PATH=./model_files/feature_scaler_final.pkl
LABELS_PATH=./model_files/label_mapping.json
CHROMA_DB_DIR=./chroma_db
```

---

## API Endpoints

| Method | Endpoint | Input | Output |
|---|---|---|---|
| GET | `/` | none | health check |
| POST | `/api/predict` | audio file (multipart/form-data) | species, confidence, explanation, image_url, metadata |
| POST | `/api/ask` | question, species (optional), history (optional) | answer string |
| GET | `/api/species` | none | list of all 50 species with metadata |
| GET | `/api/species/{name}` | species name in URL | full species info + description |

---

## POST /api/predict — Full Spec

### Input
- Content-Type: `multipart/form-data`
- Field name: `audio`
- Accepted formats: `.mp3`, `.wav`, `.m4a`, `.ogg`, `.webm`
- Max file size: 10MB

### Processing Steps (in order)
1. Validate file format and size
2. Convert audio to 48kHz mono WAV using librosa
3. Normalise amplitude to [-1, 1]
4. Split into 3-second chunks
5. Extract 6,522-dimensional BirdNET embeddings per chunk
6. Apply StandardScaler (`feature_scaler_final.pkl`)
7. Run MLP classifier (`singapore_bird_classifier_final.pth`)
8. Aggregate predictions across chunks — return highest confidence species
9. Call `get_bird_explanation(species, confidence)` from rag.py
10. Call `get_bird_image_url(species)` for Wikipedia image

### Output JSON
```json
{
  "species": "Collared Kingfisher",
  "confidence": 0.87,
  "explanation": "The Collared Kingfisher is...",
  "image_url": "https://upload.wikimedia.org/...",
  "metadata": {
    "family": "Kingfishers (Alcedinidae)",
    "malay_name": "Pekaka Bakau Biasa",
    "chinese_name": "白领翡翠",
    "status": "Res",
    "status_full": "Resident",
    "rdb3": "LC",
    "rdb3_full": "Least Concern"
  }
}
```

### Error Responses
- `400` — unsupported file format or file too large
- `422` — no audio file provided
- `500` — model inference failed
- If confidence < 0.30, still return result but include `"low_confidence": true` in response

---

## POST /api/ask — Full Spec

### Input JSON
```json
{
  "question": "Where can I spot this bird?",
  "species": "Collared Kingfisher",
  "history": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```
- `species` is optional — omit for general questions
- `history` is optional — include for conversational follow-up

### Smart Query Routing Logic
| Question contains | Action |
|---|---|
| "migratory", "migrant", "migrate" | Filter ChromaDB by `status = Mi` |
| "introduced", "non-native" | Filter ChromaDB by `status = Intr` |
| "resident", "year-round" | Filter ChromaDB by `status = Res` |
| "endangered", "threatened", "vulnerable", "conservation" | Filter ChromaDB by `rdb3` in `[VU, EN, CR, NT]` |
| anything else | Semantic similarity search, top_k=8, optional species filter |
| history provided | Use conversational mode with session context |

### Output JSON
```json
{
  "answer": "The Collared Kingfisher can be found..."
}
```

---

## GET /api/species — Full Spec

Returns list of all 50 species with their metadata from ChromaDB.

### Output JSON
```json
[
  {
    "species": "Collared Kingfisher",
    "family": "Kingfishers (Alcedinidae)",
    "malay_name": "Pekaka Bakau Biasa",
    "chinese_name": "白领翡翠",
    "status": "Res",
    "status_full": "Resident",
    "rdb3": "LC",
    "rdb3_full": "Least Concern",
    "image_url": "https://upload.wikimedia.org/..."
  }
]
```

---

## app/config.py

```python
import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
MODEL_PATH        = os.getenv("MODEL_PATH", "./model_files/singapore_bird_classifier_final.pth")
SCALER_PATH       = os.getenv("SCALER_PATH", "./model_files/feature_scaler_final.pkl")
LABELS_PATH       = os.getenv("LABELS_PATH", "./model_files/label_mapping.json")
CHROMA_DB_DIR     = os.getenv("CHROMA_DB_DIR", "./chroma_db")
```

---

## app/services/rag.py — Functions to Implement

Implement these functions. Load embedder, ChromaDB client, and Claude client once at module level (not per request).

```python
# Load at module level
embedder   = SentenceTransformer("all-MiniLM-L6-v2")
chroma     = chromadb.PersistentClient(path=CHROMA_DB_DIR)
collection = chroma.get_collection("singapore_birds")
claude     = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
```

### retrieve(query, species_filter=None, top_k=8)
- Encode query with embedder
- Query ChromaDB with optional species metadata filter
- Return list of dicts: `[{"text": ..., "metadata": ..., "distance": ...}]`

### get_species_by_status(status)
- Query ChromaDB directly by metadata field `status == status`
- Deduplicate by species name (one entry per species)
- Return list of dicts: `[{"text": ..., "metadata": ...}]`

### get_bird_explanation(species, confidence)
- Retrieve chunks filtered to species
- Build prompt with species metadata + context
- Call Claude API
- Return explanation string (150–200 words)
- Prompt must include: physical description, Singapore locations, resident/migratory status, conservation status, interesting facts

### ask_about_bird(question, species=None)
- Apply smart query routing logic (see POST /api/ask routing table above)
- Build context from retrieved chunks
- Build metadata summary string for all retrieved species
- Call Claude API
- Return answer string (under 200 words)

### ask_about_bird_conversational(question, species, history)
- Retrieve chunks filtered to species, top_k=6
- Build system prompt with species metadata
- Append new question + context to history
- Call Claude API with full message history
- Return tuple: (answer_string, updated_history_list)

### get_bird_image_url(species_name, latin_name="")
- Call Wikipedia REST API: `https://en.wikipedia.org/api/rest_v1/page/summary/{species_name}`
- If no thumbnail found, try latin_name as fallback
- Return image URL string or empty string if not found
- Wrap in try/except, timeout=5

---

## app/models/classifier.py — BirdClassifier Class

> ⚠️ Do not implement the BirdNET embedding extraction logic. Leave a placeholder stub that accepts audio_path and returns dummy predictions. The actual BirdNET inference code will be provided separately.

```python
class BirdClassifier:
    def __init__(self):
        # Load label_mapping.json
        # Load feature_scaler_final.pkl
        # Load singapore_bird_classifier_final.pth (MLP weights)
        # TODO: BirdNET embedding extraction will be added separately
        pass

    def predict(self, audio_path: str) -> dict:
        # TODO: implement full inference pipeline
        # For now return a stub response for testing
        return {
            "species": "Collared Kingfisher",
            "confidence": 0.87
        }
```

---

## app/services/audio.py

```python
def preprocess_audio(file_bytes: bytes, filename: str) -> str:
    # Save uploaded bytes to a temp file
    # Convert to 48kHz mono WAV using librosa
    # Normalise amplitude to [-1, 1]
    # Return path to processed temp WAV file
    pass

def validate_audio(filename: str, file_size: int) -> bool:
    # Check extension is in [.mp3, .wav, .m4a, .ogg, .webm]
    # Check file_size <= 10MB
    # Return True if valid, raise HTTPException if not
    pass
```

---

## app/api/routes.py

```python
router = APIRouter()

@router.post("/predict")
async def predict(audio: UploadFile = File(...)):
    # 1. validate_audio()
    # 2. preprocess_audio()
    # 3. classifier.predict()
    # 4. get_bird_explanation()
    # 5. get_bird_image_url()
    # 6. return full response JSON

@router.post("/ask")
async def ask(body: AskRequest):
    # Route to ask_about_bird() or ask_about_bird_conversational()
    # based on whether history is provided

@router.get("/species")
async def get_all_species():
    # Query ChromaDB for one chunk per species
    # Fetch image URLs for each
    # Return list

@router.get("/species/{name}")
async def get_species(name: str):
    # Query ChromaDB filtered to species name
    # Return full info + image URL
```

---

## ChromaDB Collection Details

- Collection name: `singapore_birds`
- Total chunks: 91
- Embedding model: `all-MiniLM-L6-v2`

### Metadata fields per chunk
| Field | Type | Example |
|---|---|---|
| `species` | string | `Collared Kingfisher` |
| `page_num` | int | `13` |
| `content_type` | string | `species` |
| `family` | string | `Kingfishers (Alcedinidae)` |
| `malay_name` | string | `Pekaka Bakau Biasa` |
| `chinese_name` | string | `白领翡翠` |
| `status` | string | `Res` |
| `status_full` | string | `Resident` |
| `rdb3` | string | `LC` |
| `rdb3_full` | string | `Least Concern` |

---

## Frontend Requirements

### Pages

#### Page 1: Home / Identify (main page)
- Hero section: tagline "Discover Singapore's Birds by Sound"
- Audio upload zone: drag-and-drop or click, accepts .mp3 .wav .m4a .ogg .webm
- Record button: triggers microphone with animated waveform
- Loading spinner while API processes
- Results card (shown after detection):
  - Species name (large) + confidence bar
  - Bird photo from Wikipedia (placeholder silhouette SVG if not found)
  - Status badge + RDB3 badge (colour coded)
  - Malay name, Chinese name, Family
  - AI explanation (2–3 paragraphs)
- Chat interface below results:
  - Suggested question chips: "Where can I spot this?", "What does it eat?", "Is it endangered?", "Tell me more"
  - Chat bubbles (user right, AI left)
  - Input box + send button
  - Species passed automatically with every message

#### Page 2: Species Browser
- Responsive grid: 2 cols mobile / 3 cols tablet / 4 cols desktop
- Each card: photo, common name, Malay name, status badge, RDB3 badge
- Filter buttons: All / Resident / Migrant / Introduced / Vagrant
- Search bar: filter by name or family
- Click card → species detail modal with full info

#### Page 3: Ask a Question
- Full-page chat interface
- Suggested chips: "Which birds are migratory?", "Which birds are endangered?", "Birds found in mangroves?", "Which birds are introduced?"
- Chat bubbles
- Optional species filter dropdown
- Input + send at bottom

### Component Structure
```
src/
├── pages/
│   ├── Home.jsx
│   ├── Species.jsx
│   └── Ask.jsx
├── components/
│   ├── AudioUpload.jsx
│   ├── ResultsCard.jsx
│   ├── ChatInterface.jsx
│   ├── SpeciesCard.jsx
│   ├── SpeciesModal.jsx
│   ├── StatusBadge.jsx
│   └── Navbar.jsx
├── api/
│   └── api.js
└── App.jsx
```

### api.js Functions
```javascript
predictBird(audioFile)                    // POST /api/predict
askQuestion(question, species, history)   // POST /api/ask
getAllSpecies()                            // GET /api/species
getSpeciesDetail(name)                    // GET /api/species/{name}
```

### Badge Colour Coding
| Badge | Value | Colour |
|---|---|---|
| Status | Resident | Green |
| Status | Migrant | Blue |
| Status | Introduced | Orange |
| Status | Vagrant | Grey |
| RDB3 | LC | Green |
| RDB3 | NT | Yellow |
| RDB3 | VU | Orange |
| RDB3 | EN | Red |
| RDB3 | CR | Dark Red |

### Frontend Environment Variable
```
VITE_API_URL=http://localhost:8000
```

---

## Non-Functional Requirements

- Audio prediction: < 10 seconds end-to-end
- RAG explanation: < 5 seconds
- Chat response: < 5 seconds
- File upload max: 10MB
- CORS: allow localhost:5173 in dev, frontend domain in production
- API key: never exposed to frontend, always via .env
- Error states: always show user-friendly message, never blank screen

---

## Development Phases

| Phase | What to build | Done when |
|---|---|---|
| 1 | Folder structure + main.py + config.py + requirements.txt | `uvicorn app.main:app --reload` runs, GET / returns health check |
| 2 | audio.py + classifier.py stub + /api/predict endpoint | POST /api/predict returns stub JSON with dummy species |
| 3 | rag.py + /api/ask + /api/species endpoints | All endpoints return correct responses via /docs |
| 4 | React frontend: AudioUpload + ResultsCard + routing | Upload audio → see result card |
| 5 | Species browser page | Grid renders, filters work, modal opens |
| 6 | Ask page + conversational chat | Follow-up questions work with session history |
| 7 | Error states + loading states + polish | No blank screens on errors, deploy-ready |

---

## Instructions for Claude Code

### Start with Phase 1 only
1. Create the full folder structure from the Project Structure section above
2. Create `app/main.py` — FastAPI app, CORS middleware (allow localhost:5173), health check at GET /
3. Create `app/config.py` — load all env vars
4. Create `requirements.txt` — all backend dependencies
5. Stop and wait for confirmation before Phase 2

### Rules
- Do NOT modify or regenerate: `model_files/`, `chroma_db/`, `.env`
- Do NOT implement BirdNET embedding extraction — leave as stub in classifier.py
- Do NOT proceed to next phase without confirmation
- Load all ML models and clients at module level, not per request
- All secrets via environment variables only
- Python: type hints on all functions
- React: functional components + hooks only, Tailwind classes only, no inline styles
- All API calls in `src/api/api.js` only, never directly in components
