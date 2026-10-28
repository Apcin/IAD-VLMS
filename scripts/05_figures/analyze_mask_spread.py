"""
Analyze Real-IAD masks to determine if defect regions are contiguous or scattered.
This helps validate if a single centroid is a sufficient representation of location.
"""
import json
import logging
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.ndimage import label, center_of_mass, distance_transform_edt
import config_mmad as config
import utils_realiad as utils

# Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Config
REALIAD_ROOT = config.DATA_DIR / "Real-IAD"
IMAGES_ROOT = REALIAD_ROOT / "images"
JSON_ROOT = REALIAD_ROOT / "json" / "realiad_jsons"

def analyze_mask_spread(mask_path: Path):
    """
    Analyze a mask to check if components are close or spread out.
    Returns:
        num_components: Number of distinct connected components
        max_distance: Maximum distance between centroids of components (normalized by image diagonal)
        is_spread: Boolean, True if components are significantly spread out
    """
    try:
        # Open and binary threshold
        img = Image.open(mask_path).convert('L')
        arr = np.array(img)
        binary = arr > 128
        
        if not np.any(binary):
            return 0, 0.0, False
            
        # Label connected components
        # struct defines connectivity (default 3x3 usually fine for 8-connectivity)
        labeled, n_components = label(binary)
        
        if n_components <= 1:
            return n_components, 0.0, False
            
        # Calculate centroids of all components
        centroids = center_of_mass(binary, labeled, range(1, n_components + 1))
        
        # If multiple components, calculate max pairwise distance
        max_dist = 0.0
        
        # Image diagonal for normalization
        h, w = arr.shape
        diag = np.sqrt(h**2 + w**2)
        
        centroids = np.array(centroids)
        
        if len(centroids) > 1:
            # Simple O(N^2) pairwise distance
            for i in range(len(centroids)):
                for j in range(i + 1, len(centroids)):
                    d = np.linalg.norm(centroids[i] - centroids[j])
                    if d > max_dist:
                        max_dist = d
        
        normalized_dist = max_dist / diag
        
        # Threshold for "spread out"
        # If components are more than 20% of image diagonal apart, consider them spread
        is_spread = normalized_dist > 0.2
        
        return n_components, normalized_dist, is_spread
        
    except Exception as e:
        logger.error(f"Error processing {mask_path}: {e}")
        return 0, 0.0, False

def main():
    logger.info("Loading metadata...")
    entries = utils.load_realiad_metadata(JSON_ROOT)
    
    logger.info(f"Found {len(entries)} total entries.")
    
    # Filter for entries with masks
    mask_entries = [e for e in entries if e.get('mask_rel_path')]
    logger.info(f"Found {len(mask_entries)} entries with defined masks.")
    
    total_masks = 0
    single_component_masks = 0
    multi_component_close = 0
    multi_component_spread = 0
    
    # Details for inspection
    spread_examples = []
    
    logger.info("Analyzing masks...")
    for entry in tqdm(mask_entries):
        mask_path = IMAGES_ROOT / entry['mask_rel_path']
        
        if not mask_path.exists():
            # Try extensions
            found = False
            for ext in ['.png', '.jpg', '.jpeg']:
                mp = mask_path.with_suffix(ext)
                if mp.exists():
                    mask_path = mp
                    found = True
                    break
            if not found:
                continue
                
        total_masks += 1
        n_comp, dist, is_spread = analyze_mask_spread(mask_path)
        
        if n_comp <= 1:
            single_component_masks += 1
        else:
            if is_spread:
                multi_component_spread += 1
                if len(spread_examples) < 10:
                    spread_examples.append({
                        'id': entry['image_id'],
                        'comps': n_comp,
                        'dist': f"{dist:.2f}",
                        'category': entry['category']
                    })
            else:
                multi_component_close += 1

    if total_masks == 0:
        logger.warning("No valid masks found on disk to analyze.")
        return

    logger.info("=== Analysis Results ===")
    logger.info(f"Total Analyzed Masks: {total_masks}")
    logger.info(f"Single Component (Continuous): {single_component_masks} ({single_component_masks/total_masks:.1%})")
    logger.info(f"Multi-Component (Close together): {multi_component_close} ({multi_component_close/total_masks:.1%})")
    logger.info(f"Multi-Component (Spread out > 20% diag): {multi_component_spread} ({multi_component_spread/total_masks:.1%})")
    
    logger.info("")
    logger.info("Summary:")
    logger.info(f"Safe for Centroid Logic (Single or Close): {single_component_masks + multi_component_close} ({(single_component_masks + multi_component_close)/total_masks:.1%})")
    logger.info(f"Potentially Problematic (Spread): {multi_component_spread} ({multi_component_spread/total_masks:.1%})")
    
    # Save results to JSON for Streamlit app
    results_path = config.DATA_DIR / "output" / "mask_analysis_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        'spread_mask_ids': [e['id'] for e in spread_examples] + [e['id'] for e in mask_entries if e['id'] not in [ex['id'] for ex in spread_examples] and analyze_mask_spread(IMAGES_ROOT / e['mask_rel_path'])[2]], 
        'analysis_summary': {
            'total_masks': total_masks,
            'single_component': single_component_masks,
            'multi_close': multi_component_close,
            'multi_spread': multi_component_spread
        },
        'mask_details': {}
    }
    
    # Re-run full collection for file saving (or refactor above loop to store all)
    # Let's refactor slightly to save time since we just ran it? 
    # Actually, the user said "run the script once", so we should save everything now.
    # The loop above only saved examples. Let's re-run or better yet, store in the loop.
    pass 

def analyze_and_save():
    logger.info("Loading metadata...")
    entries = utils.load_realiad_metadata(JSON_ROOT)
    mask_entries = [e for e in entries if e.get('mask_rel_path')]
    
    results = {}
    
    logger.info("Analyzing masks and saving results...")
    for entry in tqdm(mask_entries):
        mask_path = IMAGES_ROOT / entry['mask_rel_path']
        
        # Handle extensions
        if not mask_path.exists():
            found = False
            for ext in ['.png', '.jpg', '.jpeg']:
                mp = mask_path.with_suffix(ext)
                if mp.exists():
                    mask_path = mp
                    found = True
                    break
            if not found: continue
            
        n_comp, dist, is_spread = analyze_mask_spread(mask_path)
        
        results[entry['image_id']] = {
            'num_components': int(n_comp),
            'max_dist_normalized': float(dist),
            'is_spread': bool(is_spread),
            'mask_type': 'Spread' if is_spread else ('Multi-Close' if n_comp > 1 else 'Single')
        }
        
        # Incremental save every 500 items to avoid data loss on crash
        if len(results) % 500 == 0:
            try:
                temp_file = config.DATA_DIR / "output" / "mask_analysis_temp.json"
                temp_file.parent.mkdir(parents=True, exist_ok=True)
                with open(temp_file, 'w') as f:
                    json.dump(results, f, indent=2)
            except Exception as e:
                logger.warning(f"Failed to save temp file: {e}")
        
    # Save
    out_file = config.DATA_DIR / "output" / "mask_analysis_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved analysis for {len(results)} masks to {out_file}")

if __name__ == "__main__":
    # main() # Replace main with the saving version
    analyze_and_save()

