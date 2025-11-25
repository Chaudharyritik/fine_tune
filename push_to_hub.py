import os
from huggingface_hub import HfApi, create_repo, login

def push_to_hub():
    print("🚀 Preparing to push model to Hugging Face Hub...")
    
    # Configuration
    local_folder = "./whisper-hindi-ct2"
    target_repo = "chaudharyritik1/whisper-hindi-v1"
    target_subfolder = "Models/whisper-hindi-ct2"
    
    # 1. Login
    print("\n🔑 Authentication")
    try:
        api = HfApi()
        user = api.whoami()
        print(f"✅ Logged in as: {user['name']}")
    except:
        login()
        api = HfApi()
        user = api.whoami()

    print(f"\n🎯 Target Repository: {target_repo}")
    print(f"📂 Target Folder: {target_subfolder}")
    
    # 2. Upload Files
    print(f"\n📤 Uploading files from '{local_folder}' to '{target_subfolder}'...")
    
    try:
        api.upload_folder(
            folder_path=local_folder,
            repo_id=target_repo,
            repo_type="model",
            path_in_repo=target_subfolder,
            commit_message="Upload CTranslate2 converted model to Models folder"
        )
        print("\n" + "="*50)
        print("✅ Upload Complete!")
        print("="*50)
        print(f"\nYour model files are now at:")
        print(f"https://huggingface.co/{target_repo}/tree/main/{target_subfolder}")
        
        print(f"\n⚠️  NOTE: Because the model is in a subfolder, you cannot load it directly with WhisperModel('repo_id').")
        print(f"You will need to download the folder first or use the local path if you clone the repo.")
        
    except Exception as e:
        print(f"\n❌ Upload failed: {e}")

if __name__ == "__main__":
    push_to_hub()
