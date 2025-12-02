import os
from huggingface_hub import HfApi, create_repo, login

def push_to_hub():
    print("🚀 Preparing to push model to Hugging Face Hub...")
    
    # Configuration
    local_folder = "./whisper-hindi-cv-merged"
    target_repo = "rishi-dbee/dbee-stt-whisper-cv"
    
    # 1. Login
    print("\n🔑 Authentication")
    token = os.getenv("HF_TOKEN")
    try:
        if token:
            login(token=token)
            api = HfApi(token=token)
        else:
            api = HfApi()
            
        user = api.whoami()
        print(f"✅ Logged in as: {user['name']}")
    except Exception as e:
        print(f"⚠️ Login failed: {e}")
        print("Tip: Run 'huggingface-cli login' or set HF_TOKEN environment variable.")
        return

    # 2. Create Repo
    print(f"\n🎯 Target Repository: {target_repo}")
    try:
        create_repo(target_repo, repo_type="model", exist_ok=True)
        print("✅ Repository ready.")
    except Exception as e:
        print(f"⚠️ Could not create repo (might exist or permission issue): {e}")

    # 3. Upload Files
    print(f"\n📤 Uploading files from '{local_folder}'...")
    
    try:
        api.upload_folder(
            folder_path=local_folder,
            repo_id=target_repo,
            repo_type="model",
            commit_message="Upload fine-tuned Whisper model on Common Voice Hindi"
        )
        print("\n" + "="*50)
        print("✅ Upload Complete!")
        print("="*50)
        print(f"\nYour model is available at:")
        print(f"https://huggingface.co/{target_repo}")
        
    except Exception as e:
        print(f"\n❌ Upload failed: {e}")

if __name__ == "__main__":
    push_to_hub()
