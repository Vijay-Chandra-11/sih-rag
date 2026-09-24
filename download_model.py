from sentence_transformers import SentenceTransformer

print("Starting model download... This may take a while depending on your network speed.")

# This line will download and cache the model
model = SentenceTransformer("all-MiniLM-L6-v2")

print("\n✅ Model download complete and cached successfully!")