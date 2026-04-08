import os
import zipfile
import shutil

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOWNLOADS_DIR = os.path.join(os.path.expanduser("~"), "Downloads")
ZIP_PATH = os.path.join(DOWNLOADS_DIR, "tess dataset.zip")
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw")
TMP_EXTRACT = os.path.join(BASE_DIR, "data", "tmp_tess")

# Map exactly how TESS folders are named into our emotion names
# TESS folders are typically like: OAF_angry, YAF_sad, etc.
EMOTION_MAP = {
    "angry": "angry",
    "happy": "happy",
    "sad": "sad",
    "neutral": "neutral"
}

def import_tess():
    if not os.path.exists(ZIP_PATH):
        print(f"ERROR: Could not find {ZIP_PATH}")
        return

    print("Extracting TESS dataset...")
    os.makedirs(TMP_EXTRACT, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(TMP_EXTRACT)

    print("Moving files into mapped emotion directories...")
    moved = 0
    # Walk through the extracted files
    for root, dirs, files in os.walk(TMP_EXTRACT):
        for file in files:
            if file.endswith(".wav"):
                file_lower = file.lower()
                
                # Check which emotion it belongs to based on the file name or folder name
                # TESS files look like OAF_back_angry.wav
                target_emotion = None
                for keyword, target_dir in EMOTION_MAP.items():
                    if f"_{keyword}" in file_lower or f"{keyword}" in root.lower():
                        target_emotion = target_dir
                        break

                if target_emotion:
                    src_file = os.path.join(root, file)
                    dest_folder = os.path.join(RAW_DATA_PATH, target_emotion)
                    os.makedirs(dest_folder, exist_ok=True)
                    dest_file = os.path.join(dest_folder, file)
                    
                    if not os.path.exists(dest_file):
                        shutil.copy2(src_file, dest_file)
                        moved += 1

    print(f"Successfully copied {moved} TESS files!")
    
    # Cleanup
    shutil.rmtree(TMP_EXTRACT)
    print("Cleanup complete. TESS is now integrated into training pipeline!")

if __name__ == "__main__":
    import_tess()
