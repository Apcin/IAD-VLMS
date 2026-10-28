import os
import zipfile
from pathlib import Path
from huggingface_hub import snapshot_download
from tqdm import tqdm

def download_and_setup_mmad():
    # Setup paths
    current_dir = Path(__file__).parent
    data_dir = current_dir / "data"
    mmad_dir = data_dir / "MMAD"
    
    print(f"Setting up MMAD dataset in: {mmad_dir}")
    mmad_dir.mkdir(parents=True, exist_ok=True)
    
    # Download from Hugging Face
    print("Downloading from Hugging Face (jiang-cc/MMAD)...")
    repo_id = "jiang-cc/MMAD"
    
    # We download to a cache dir first or directly to our target?
    # snapshot_download downloads to cache by default and returns path.
    # But we want the files in our specific directory.
    # We can use local_dir to download directly there.
    
    download_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=mmad_dir,
        # local_dir_use_symlinks=False, # Deprecated
        # resume_download=True # Deprecated, default behavior
    )
    
    print(f"Downloaded files to {download_path}")
    
    # Extract zip files
    zip_files = list(mmad_dir.glob("*.zip"))
    print(f"Found {len(zip_files)} zip files to extract.")
    
    for zip_path in tqdm(zip_files, desc="Extracting datasets"):
        print(f"Extracting {zip_path.name}...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Extract to the MMAD directory
                # If the zip contains a root folder (e.g. MVTec-AD/), it will be created inside MMAD/
                zip_ref.extractall(mmad_dir)
            
            print(f"Extracted {zip_path.name}")
            
            # Optional: Remove zip file to save space
            # os.remove(zip_path)
            # print(f"Removed {zip_path.name}")
            
        except zipfile.BadZipFile:
            print(f"Error: {zip_path.name} is not a valid zip file.")
        except Exception as e:
            print(f"Error extracting {zip_path.name}: {e}")

    print("\nMMAD Dataset setup complete!")
    print(f"Location: {mmad_dir}")
    
    # List contents to verify
    print("\nDirectory contents:")
    for item in mmad_dir.iterdir():
        if item.is_dir():
            print(f"  [DIR]  {item.name}")
        elif item.name.endswith('.json') or item.name.endswith('.csv'):
             print(f"  [FILE] {item.name}")

if __name__ == "__main__":
    download_and_setup_mmad()
