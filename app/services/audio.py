import os
import tempfile

import librosa
import numpy as np
import soundfile as sf
from fastapi import HTTPException

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".webm"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
TARGET_SR = 48_000


def validate_audio(filename: str, file_size: int) -> bool:
    ext = os.path.splitext(filename)[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Accepted: {', '.join(ALLOWED_EXTENSIONS)}",
        )
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({file_size / 1_048_576:.1f} MB). Maximum allowed is 10 MB.",
        )
    return True


def preprocess_audio(file_bytes: bytes, filename: str) -> str:
    ext = os.path.splitext(filename)[-1].lower()

    # Write raw upload to a temp file so librosa can read it
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as raw_tmp:
        raw_tmp.write(file_bytes)
        raw_tmp_path = raw_tmp.name

    try:
        # Load as mono at 48 kHz
        audio, _ = librosa.load(raw_tmp_path, sr=TARGET_SR, mono=True)
    finally:
        os.unlink(raw_tmp_path)

    # Normalise amplitude to [-1, 1]
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val

    # Write processed WAV to a new temp file (caller is responsible for cleanup)
    out_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    out_tmp.close()
    sf.write(out_tmp.name, audio, TARGET_SR, subtype="PCM_16")

    return out_tmp.name
