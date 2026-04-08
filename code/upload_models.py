import os
import glob
from huggingface_hub import HfApi, create_repo

# ================= KONFIGURASI =================
HF_USERNAME = "Pataangg" # GANTI DENGAN USERNAME ANDA
FE_REPO_NAME = f"{HF_USERNAME}/whisper-minang-freeze-encoder"
LORA_REPO_NAME = f"{HF_USERNAME}/whisper-minang-lora"

base_dir = os.path.dirname(os.path.abspath(__file__))
fe_checkpoints_dir = os.path.join(base_dir, "freeze_encoder/outputs/checkpoints")
lora_checkpoints_dir = os.path.join(base_dir, "lora/outputs/checkpoints")
# ===============================================

api = HfApi()

def upload_checkpoints(local_dir, repo_id):
    print(f"\n🚀 Memulai upload ke repository: {repo_id}")
    
    # Buat repo jika belum ada di Hugging Face
    create_repo(repo_id=repo_id, exist_ok=True, repo_type="model", private=False)
    
    # Cari folder fold_0 sampai fold_4
    for fold in range(5):
        fold_name = f"fold_{fold}"
        fold_path = os.path.join(local_dir, fold_name)
        
        if not os.path.exists(fold_path):
            print(f"⚠️ Melewati {fold_name}: Folder tidak ditemukan.")
            continue
            
        # Cari folder checkpoint-xxx terbaik di dalam fold
        checkpoints = glob.glob(os.path.join(fold_path, "checkpoint-*"))
        if not checkpoints:
            print(f"⚠️ Melewati {fold_name}: Tidak ada checkpoint di dalam folder ini.")
            continue
            
        # Asumsikan checkpoint dengan angka terbesar adalah yang terakhir/terbaik
        best_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))[-1]
        
        print(f"Mengupload {os.path.basename(best_checkpoint)} ke {repo_id} (sebagai {fold_name})...")
        try:
            api.upload_folder(
                folder_path=best_checkpoint,
                repo_id=repo_id,
                repo_type="model",
                path_in_repo=fold_name  # Jadikan subfolder bernama fold_0, fold_1, dst.
            )
            print(f"✅ Berhasil upload {fold_name}!")
        except Exception as e:
            print(f"❌ Gagal upload {fold_name}: {e}")

if __name__ == "__main__":
    print("Mulai proses upload model...")
    # 1. Upload Freeze Encoder
    if os.path.exists(fe_checkpoints_dir):
        upload_checkpoints(fe_checkpoints_dir, FE_REPO_NAME)
    else:
        print(f"Direktori {fe_checkpoints_dir} tidak ditemukan.")
        
    # 2. Upload LoRA
    if os.path.exists(lora_checkpoints_dir):
        upload_checkpoints(lora_checkpoints_dir, LORA_REPO_NAME)
    else:
        print(f"Direktori {lora_checkpoints_dir} tidak ditemukan.")
        
    print("\n🎉 Proses upload selesai!")
