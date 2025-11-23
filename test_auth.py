from datasets import load_dataset
from huggingface_hub import whoami

try:
    user = whoami()
    print(f"Authenticated as: {user['name']}")
except Exception as e:
    print(f"Auth check failed: {e}")

print("Attempting to load google/fleurs (Hindi)...")
try:
    ds = load_dataset("google/fleurs", "hi_in", split="test", trust_remote_code=True)
    print("Successfully loaded FLEURS!")
except Exception as e:
    print(f"Failed to load FLEURS: {e}")

print("Attempting to load ai4bharat/svarah...")
try:
    ds = load_dataset("ai4bharat/svarah", split="train", trust_remote_code=True, token=True)
    print("Successfully loaded Svarah!")
except Exception as e:
    print(f"Failed to load Svarah: {e}")
