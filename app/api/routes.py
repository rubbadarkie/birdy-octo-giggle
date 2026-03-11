import os

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.models.classifier import BirdClassifier
from app.services.audio import preprocess_audio, validate_audio

router = APIRouter()

# Load classifier once at module level
classifier = BirdClassifier()


@router.post("/predict")
async def predict(audio: UploadFile = File(...)) -> dict:
    file_bytes = await audio.read()

    # 1. Validate format and size
    validate_audio(audio.filename or "", len(file_bytes))

    # 2 & 3. Preprocess audio (convert to 48 kHz mono WAV, normalise)
    tmp_path: str | None = None
    try:
        tmp_path = preprocess_audio(file_bytes, audio.filename or "upload.wav")

        # 4. Run classifier (stub for now — BirdNET integration pending)
        result = classifier.predict(tmp_path)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {exc}") from exc
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    species = result["species"]
    confidence = result["confidence"]

    response: dict = {
        "species": species,
        "confidence": confidence,
        "explanation": "",   # populated in Phase 3 via rag.get_bird_explanation()
        "image_url": "",     # populated in Phase 3 via rag.get_bird_image_url()
        "metadata": {},      # populated in Phase 3 from ChromaDB metadata
    }

    if confidence < 0.30:
        response["low_confidence"] = True

    return response
