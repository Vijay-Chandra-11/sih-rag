from sentence_transformers import SentenceTransformer
from TTS.api import TTS

print("--- Starting Pre-download of All Required AI Models ---")
print("This may take a very long time depending on your network. Please be patient.")

# --- 1. Embedding Model ---
try:
    print("\n[1/2] Downloading Embedding Model (all-MiniLM-L6-v2)...")
    SentenceTransformer("all-MiniLM-L6-v2")
    print("✅ Embedding Model is ready.")
except Exception as e:
    print(f"❌ Error downloading embedding model: {e}")

# --- 2. Text-to-Speech Model ---
try:
    print("\n[2/2] Downloading TTS Model (this is a large file)...")
    TTS("tts_models/en/ljspeech/tacotron2-DDC")
    print("✅ TTS Model is ready.")
except Exception as e:
    print(f"❌ Error downloading TTS model: {e}")


print("\n--- All models have been downloaded and cached successfully! ---")