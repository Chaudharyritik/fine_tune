import os
from huggingface_hub import HfApi, login

def push_ct2_to_hub():
    print("🚀 Preparing to push CT2 model to Hugging Face Hub...")
    
    # Configuration
    local_folder = "./whisper-hindi-cv-ct2"
    target_repo = "rishi-dbee/dbee-stt-whisper-cv"
    path_in_repo = "ct2" # Upload to a 'ct2' subfolder to avoid overwriting PyTorch model
    
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

    # 2. Upload Files
    print(f"\n📤 Uploading files from '{local_folder}' to '{target_repo}/{path_in_repo}'...")
    
    try:
        api.upload_folder(
            folder_path=local_folder,
            repo_id=target_repo,
            repo_type="model",
            path_in_repo=path_in_repo,
            commit_message="Upload CTranslate2 converted model"
        )
        print("\n" + "="*50)
        print("✅ Upload Complete!")
        print("="*50)
        print(f"\nYour CT2 model is available at:")
        print(f"https://huggingface.co/{target_repo}/tree/main/{path_in_repo}")
        
    except Exception as e:
        print(f"\n❌ Upload failed: {e}")

if __name__ == "__main__":
    push_ct2_to_hub()
