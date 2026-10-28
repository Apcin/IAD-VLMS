"""
Script to download Real-IAD dataset from Hugging Face.
"""
import os
import sys
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm

# Configuration
HF_TOKEN = os.environ.get("HF_TOKEN", "")  # set via `export HF_TOKEN=...` or `huggingface-cli login`
REPO_ID = "Real-IAD/Real-IAD"
DATA_ROOT = Path(__file__).parent / "data" / "Real-IAD"
IMAGES_DIR = DATA_ROOT / "images"
JSON_DIR = DATA_ROOT / "json"

def setup_directories():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    JSON_DIR.mkdir(parents=True, exist_ok=True)

def download_file_direct(url, output_path):
    if output_path.exists():
        print(f"File already exists: {output_path}, checking integrity...")
        # Optional: Check size or hash? For now just skip if exists to save time/bandwidth
        print(f"Skipping download for {output_path.name}")
        return True

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    try:
        response = requests.get(url, headers=headers, stream=True)
        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))
            with open(output_path, 'wb') as f, tqdm(
                desc=output_path.name,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in response.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    bar.update(size)
            return True
        else:
            print(f"Failed to download {url}: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def download_jsons():
    print("Downloading metadata (JSONs)...")
    url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/realiad_jsons.zip"
    zip_path = DATA_ROOT / "realiad_jsons.zip"
    
    # Check if already extracted
    if (JSON_DIR / "realiad_jsons").exists():
        print("Metadata already extracted.")
        return

    if download_file_direct(url, zip_path):
        print("Extracting JSONs...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(JSON_DIR)
            print("Metadata extracted.")
        except zipfile.BadZipFile:
            print("Error: Downloaded file is not a valid zip file.")

def download_images(resolution="realiad_1024", categories=None):
    """
    Download images.
    """
    print(f"Downloading images ({resolution})...")
    
    if categories is None:
        print("Please specify categories list.")
        return

    for cat in categories:
        # Check if category folder already exists and is populated
        cat_dir = IMAGES_DIR / cat
        if cat_dir.exists() and any(cat_dir.iterdir()):
            print(f"Category '{cat}' already exists. Skipping download.")
            continue

        filename = f"{cat}.zip"
        url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{resolution}/{filename}"
        zip_path = DATA_ROOT / filename
        
        print(f"Downloading {cat}...")
        if download_file_direct(url, zip_path):
            print(f"Extracting {cat}...")
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(IMAGES_DIR)
                
                # Cleanup zip to save space
                print(f"Removing {zip_path} to save space...")
                os.remove(zip_path)
                
            except Exception as e:
                print(f"Error processing {cat}: {e}")

if __name__ == "__main__":
    setup_directories()
    
    # 1. Download Metadata
    download_jsons()
    
    # 2. Download Images
    # Full list of categories
    ALL_CATEGORIES = [
        'audiojack', 'bottle_cap', 'button_battery', 'end_cap', 'eraser', 
        'fire_hood', 'mint', 'mounts', 'pcb', 'phone_battery', 
        'plastic_nut', 'plastic_plug', 'porcelain_doll', 'regulator', 
        'rolled_strip_base', 'sim_card_set', 'switch', 'tape', 
        'terminalblock', 'toothbrush', 'toy', 'toy_brick', 
        'transistor1', 'u_block', 'usb', 'usb_adaptor', 'vcpill', 
        'wooden_beads', 'woodstick', 'zipper'
    ]
    
    print(f"\n[INFO] Downloading ALL categories: {len(ALL_CATEGORIES)} total.")
    
    download_images(resolution="realiad_1024", categories=ALL_CATEGORIES)
    
    print("\nDownload complete.")
