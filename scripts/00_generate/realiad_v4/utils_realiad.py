"""
Utility functions for Real-IAD trace generation
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

# Official Real-IAD Anomaly Code Mapping
ANOMALY_MAP = {
    "OK": "Normal",
    "AK": "Pit",
    "BX": "Deformation",
    "CH": "Abrasion",
    "HS": "Scratch",
    "PS": "Damage",
    "QS": "Missing Parts",
    "YW": "Foreign Objects",
    "ZW": "Contamination",
}

def get_defect_name(code: str) -> str:
    """Convert code (BX) to readable name (Deformation)"""
    return ANOMALY_MAP.get(code, f"Anomaly {code}")

def get_mask_location(mask_path: Path) -> str:
    """
    Analyze a binary mask image to determine the defect location string.
    Supports multiple spread-out components.
    """
    try:
        from scipy.ndimage import label, center_of_mass
        
        if not mask_path.exists():
            return None
            
        # Open and convert to numpy
        img = Image.open(mask_path).convert('L')
        arr = np.array(img)
        binary = arr > 128
        
        if not np.any(binary):
            return None
            
        # Label connected components
        labeled, n_components = label(binary)
        
        if n_components == 0:
            return None
            
        # Get centroids
        centroids = center_of_mass(binary, labeled, range(1, n_components + 1))
        
        # Standardize to list of tuples (scipy returns list for sequence input, but safety first)
        if not isinstance(centroids, list):
            centroids = [centroids]
            
        # Limit to top 3 components by size if many
        if n_components > 3:
            # Calculate sizes
            sizes = [np.sum(labeled == i) for i in range(1, n_components + 1)]
            # Sort by size descending
            sorted_indices = np.argsort(sizes)[::-1][:3]
            centroids = [centroids[i] for i in sorted_indices]
            
        locations = []
        height, width = arr.shape
        
        for point in centroids:
            # Handle both single tuple (y, x) and list of tuples cases
            if isinstance(point, (list, tuple)) and len(point) == 2:
                cy, cx = point
            else:
                 # Should not happen with ndimage but safety check
                 continue

            norm_x = cx / width
            norm_y = cy / height
            
            # Determine horizontal position
            if norm_x < 0.33: h_pos = "left"
            elif norm_x > 0.66: h_pos = "right"
            else: h_pos = "center"
            
            # Determine vertical position
            if norm_y < 0.33: v_pos = "top"
            elif norm_y > 0.66: v_pos = "bottom"
            else: v_pos = "middle"
            
            loc = ""
            if v_pos == "middle":
                if h_pos == "center": loc = "center"
                else: loc = f"middle-{h_pos}"
            elif h_pos == "center":
                loc = f"{v_pos}-center"
            else:
                loc = f"{v_pos}-{h_pos}"
                
            locations.append(loc)
            
        # Remove duplicates while preserving order
        unique_locs = []
        for l in locations:
            if l not in unique_locs:
                unique_locs.append(l)
                
        if len(unique_locs) == 1:
            return unique_locs[0]
            
        return ", ".join(unique_locs)
        
    except ImportError:
        # Fallback if scipy not installed (though it should be based on other files)
        logger.warning("scipy not found, falling back to simple centroid")
        # ... original simple logic fallback ...
        try:
             # Open and convert to numpy
            img = Image.open(mask_path).convert('L')
            arr = np.array(img)
            
            # Find white pixels (> 128)
            y_indices, x_indices = np.where(arr > 128)
            
            if len(x_indices) == 0:
                return None
                
            # Calculate centroid
            mean_x = np.mean(x_indices)
            mean_y = np.mean(y_indices)
            
            # Image dimensions
            height, width = arr.shape
            
            # Normalize coordinates (0-1)
            norm_x = mean_x / width
            norm_y = mean_y / height
            
            if norm_x < 0.33: h_pos = "left"
            elif norm_x > 0.66: h_pos = "right"
            else: h_pos = "center"
            
            if norm_y < 0.33: v_pos = "top"
            elif norm_y > 0.66: v_pos = "bottom"
            else: v_pos = "middle"
            
            if v_pos == "middle":
                if h_pos == "center": return "center"
                return f"middle-{h_pos}"
            if h_pos == "center":
                return f"{v_pos}-center"
            return f"{v_pos}-{h_pos}"
        except: return None
        
    except Exception as e:
        logger.warning(f"Failed to process mask {mask_path}: {e}")
        return None

def load_realiad_metadata(json_dir: Path, categories: List[str] = None) -> List[Dict]:
    """
    Load metadata for Real-IAD dataset.
    
    Args:
        json_dir: Directory containing category JSON files
        categories: List of categories to load (if None, loads all found)
        
    Returns:
        List of image entries with:
        {
            'image_id': str,
            'image_path': str (relative to resolution root),
            'category': str,
            'ground_truth': 'Yes'/'No',
            'defect_type': str (code or name),
            'split': 'train'/'test',
            'mask_path': str (relative path from json, usually needs adjustment)
        }
    """
    entries = []
    
    if not json_dir.exists():
        logger.error(f"JSON directory not found: {json_dir}")
        return []
        
    json_files = list(json_dir.glob("*.json"))
    
    for jf in json_files:
        cat_name = jf.stem
        if categories and cat_name not in categories:
            continue
            
        try:
            with open(jf, 'r') as f:
                data = json.load(f)
            
            # Process 'test' set (usually contains both normal and anomaly)
            # 'train' is typically all normal
            
            for split in ['train', 'test']:
                if split not in data:
                    continue
                    
                for item in data[split]:
                    # Item structure:
                    # {"category": "zipper", "anomaly_class": "OK", "image_path": "...", "mask_path": ...}
                    
                    code = item['anomaly_class']
                    is_anomaly = code != 'OK'
                    
                    # Get readable defect name
                    defect_type = get_defect_name(code) if is_anomaly else "None"
                    
                    # Image path in JSON is like "OK/S0001/zipper_...jpg"
                    rel_path = Path(cat_name) / item['image_path']
                    
                    # Mask path
                    mask_rel_path = None
                    if item.get('mask_path'):
                        # Mask path in JSON usually assumes category root or similar
                        # e.g., "NG/BX/S0001/..." 
                        # We need to prepend category likely
                        mask_rel_path = str(Path(cat_name) / item['mask_path'])
                    
                    entries.append({
                        'image_id': utils_sanitize_filename(str(rel_path.with_suffix(''))),
                        'image_rel_path': str(rel_path),
                        'category': cat_name,
                        'ground_truth': 'Yes' if is_anomaly else 'No',
                        'defect_type': defect_type,
                        'split': split,
                        'mask_rel_path': mask_rel_path
                    })
                    
        except Exception as e:
            logger.error(f"Error processing {jf.name}: {e}")
            
    return entries

def utils_sanitize_filename(text: str) -> str:
    """Helper to sanitize IDs"""
    import re
    return re.sub(r'[<>:"/\\|?*]', '_', text).strip()

def create_overlay(image_path, mask_path, color=(255, 0, 0), alpha=128):
    """
    Create an overlay of the mask on the original image.
    The mask (white parts) will be colored in 'color' with transparency 'alpha'.
    """
    try:
        from PIL import Image, ImageDraw
        
        base_img = Image.open(image_path).convert("RGBA")
        mask_img = Image.open(mask_path).convert("L")
        
        # Resize mask to match base image if needed
        if base_img.size != mask_img.size:
            mask_img = mask_img.resize(base_img.size, Image.Resampling.NEAREST)
            
        # Create a solid color image
        solid_color = Image.new("RGBA", base_img.size, color + (alpha,))
        
        # Create a transparent image for the overlay
        overlay = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
        
        # Paste the solid color only where the mask is white
        overlay.paste(solid_color, (0, 0), mask_img)
        
        # Composite
        combined = Image.alpha_composite(base_img, overlay)
        
        # Draw bounding box
        bbox = mask_img.getbbox()
        if bbox:
            draw = ImageDraw.Draw(combined)
            draw.rectangle(bbox, outline="red", width=3)
            
        return combined.convert("RGB") # Convert back to RGB for API
    except Exception as e:
        logger.error(f"Error creating overlay for {image_path}: {e}")
        # Return original if overlay fails
        try:
            return Image.open(image_path).convert("RGB")
        except:
            return None
