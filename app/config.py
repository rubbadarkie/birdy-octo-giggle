import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
MODEL_PATH        = os.getenv("MODEL_PATH", "./model_files/singapore_bird_classifier_final.pth")
SCALER_PATH       = os.getenv("SCALER_PATH", "./model_files/feature_scaler_final.pkl")
LABELS_PATH       = os.getenv("LABELS_PATH", "./model_files/label_mapping.json")
CHROMA_DB_DIR     = os.getenv("CHROMA_DB_DIR", "./chroma_db")
