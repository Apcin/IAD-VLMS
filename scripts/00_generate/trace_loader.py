"""
Module to load and compile reasoning traces from MMAD dataset locally.
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Iterator
import re

try:
    import numpy as np
    from PIL import Image
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    logging.warning("numpy/PIL not available - mask location analysis will be skipped")

# Configure logging
logging.basicConfig(
    level="INFO",
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_location_from_mask(mask_path: Path) -> Optional[str]:
    """
    Analyze a binary mask image to determine the defect location string.
    Uses 3x3 grid mapping similar to Real-IAD.
    """
    try:
        if not mask_path.exists():
            return None
            
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
        
        # Determine horizontal position
        if norm_x < 0.33:
            h_pos = "left"
        elif norm_x > 0.66:
            h_pos = "right"
        else:
            h_pos = "center"
            
        # Determine vertical position
        if norm_y < 0.33:
            v_pos = "top"
        elif norm_y > 0.66:
            v_pos = "bottom"
        else:
            v_pos = "middle"
            
        # Special handling for "center"
        if h_pos == "center" and v_pos == "middle":
            return "center"
            
        # Combine vertical-horizontal
        if v_pos == "middle":
            return f"middle-{h_pos}"
        elif h_pos == "center":
            return f"{v_pos}-center"
        else:
            return f"{v_pos}-{h_pos}"
            
    except Exception as e:
        logger.warning(f"Error analyzing mask {mask_path}: {e}")
        return None

# Constants
MMAD_DIR = Path("/bulk/aacudad/reasoning_traces/reasoning_traces_gen/data/MMAD")
OUTPUT_DIR = Path("/bulk/aacudad/reasoning_traces/reasoning_traces_gen/data/output/reasoning_traces")

def clean_text(text: str) -> str:
    """Clean up text content"""
    # Remove any double spaces, newlines that break sentences awkwardly
    return re.sub(r'\s+', ' ', text).strip()

def load_visa_qa_data(mmad_dir: Path) -> Dict:
    """Load VisA and GoodsAD QA.json files to extract proper defect types and locations."""
    qa_map = {}
    
    # Load VisA QA data
    visa_dir = mmad_dir / "VisA"
    if visa_dir.exists():
        logger.info("Loading VisA QA.json files...")
        qa_map.update(_load_qa_from_directory(visa_dir, "VisA"))
    
    # Load GoodsAD QA data
    goodsad_dir = mmad_dir / "GoodsAD"
    if goodsad_dir.exists():
        logger.info("Loading GoodsAD QA.json files...")
        qa_map.update(_load_qa_from_directory(goodsad_dir, "GoodsAD"))
    
    logger.info(f"Loaded QA data for {len(qa_map)} images total")
    return qa_map

def _load_qa_from_directory(base_dir: Path, dataset_name: str) -> Dict:
    """Helper function to load QA.json files from a directory structure."""
    qa_map = {}
    
    for product_dir in base_dir.iterdir():
        if not product_dir.is_dir():
            continue
        
        qa_file = product_dir / "QA.json"
        if not qa_file.exists():
            continue
        
        try:
            with open(qa_file, 'r', encoding='utf-8') as f:
                qa_data = json.load(f)
            
            for img_rel_path, qa_entry in qa_data.items():
                conversations = qa_entry.get('conversation', [])
                
                defect_type = None
                location = None
                
                for conv in conversations:
                    conv_type = conv.get('type', '')
                    answer = conv.get('Answer', '')
                    options = conv.get('Options', {})
                    
                    # Extract defect type
                    if conv_type == 'Defect Classification' and answer in options:
                        defect_type = options[answer]
                        
                        # Handle "All of the above" - extract all actual defect types
                        if defect_type.lower() == 'all of the above':
                            # Get all other options (excluding "All of the above")
                            actual_types = [
                                opt_val for opt_val in options.values()
                                if opt_val.lower() != 'all of the above'
                            ]
                            # Join with comma
                            defect_type = ', '.join(actual_types) if actual_types else None
                    
                    # Extract location
                    elif conv_type == 'Defect Localization' and answer in options:
                        location = options[answer]
                        
                        # Handle "All of the above" for location too (though rare)
                        if location and location.lower() == 'all of the above':
                            actual_locs = [
                                opt_val for opt_val in options.values()
                                if opt_val.lower() != 'all of the above'
                            ]
                            location = ', '.join(actual_locs) if actual_locs else None
                
                # Create full relative path
                # Format: VisA/product/test/bad/000.txt or GoodsAD/product/test/broken/068_009.txt
                full_rel_path_jpg = f"{dataset_name}/{product_dir.name}/{img_rel_path}"
                full_rel_path_txt = full_rel_path_jpg.rsplit('.', 1)[0] + '.txt'
                
                qa_map[full_rel_path_txt] = {
                    'defect_type': defect_type,
                    'location': location,
                    'answer': 'Yes'  # QA.json only contains anomalies
                }
        
        except Exception as e:
            logger.error(f"Error loading QA.json for {product_dir.name}: {e}")
    
    return qa_map

def get_metadata_from_json(mmad_data: Dict, rel_path: str, visa_qa_data: Dict = None) -> Dict:
    """Extract Answer, Type, Location from mmad.json entry or VisA QA.json"""
    # Try to match the key. JSON keys use forward slash
    key = rel_path.replace("\\", "/")
    
    # Check VisA QA data first (if this is a VisA image)
    if visa_qa_data and key in visa_qa_data:
        qa_info = visa_qa_data[key]
        return {
            "answer": qa_info['answer'],
            "type": qa_info['defect_type'],
            "location": qa_info['location']
        }
    
    # Fall back to mmad.json
    if key not in mmad_data:
        # Try to find partial match if needed, or return defaults
        return None
    
    entry = mmad_data[key]
    conversation = entry.get("conversation", [])
    
    metadata = {
        "answer": "No",
        "type": None,
        "location": None
    }
    
    for turn in conversation:
        q_type = turn.get("type", "")
        ans_key = turn.get("Answer", "")
        options = turn.get("Options", {})
        answer_text = options.get(ans_key, "").strip(".").strip()
        
        if q_type == "Anomaly Detection":
            if answer_text.lower().startswith("yes"):
                metadata["answer"] = "Yes"
            else:
                metadata["answer"] = "No"
                
        elif q_type == "Defect Classification":
            metadata["type"] = answer_text
            
            # Handle "All of the above" in mmad.json too
            if answer_text and answer_text.lower() == 'all of the above':
                actual_types = [
                    opt_val for opt_val in options.values()
                    if opt_val.lower() != 'all of the above'
                ]
                metadata["type"] = ', '.join(actual_types) if actual_types else None
            
        elif q_type == "Defect Localization":
            metadata["location"] = answer_text
            
            # Handle "All of the above" for location in mmad.json too
            if answer_text and answer_text.lower() == 'all of the above':
                actual_locs = [
                    opt_val for opt_val in options.values()
                    if opt_val.lower() != 'all of the above'
                ]
                metadata["location"] = ', '.join(actual_locs) if actual_locs else None
            
    return metadata

def format_trace(image_id: str, text_content: str, metadata: Dict) -> Dict:
    """Format the trace into the target JSON structure"""
    
    # Clean the text content for the <think> block
    think_content = clean_text(text_content)
    
    # Construct the XML-like string
    xml_reasoning = f"<think>{think_content}</think>"
    
    if metadata["answer"] == "Yes":
        if metadata["location"]:
            xml_reasoning += f"\n<location>{metadata['location']}</location>"
        if metadata["type"]:
            xml_reasoning += f"\n<type>{metadata['type']}</type>"
        xml_reasoning += f"\n<answer>Yes</answer>"
    else:
        xml_reasoning += f"\n<answer>No</answer>"
        
    return {
        "image_id": image_id,
        "reasoning": xml_reasoning
    }

def load_everything(
    mmad_dir: Path = MMAD_DIR, 
    output_dir: Optional[Path] = None
) -> List[Dict]:
    """
    Load all available reasoning traces from the dataset.
    
    Args:
        mmad_dir: Path to the MMAD dataset root
        output_dir: Optional path to save the generated JSONs to disk
        
    Returns:
        List of dicts containing formatted reasoning traces
    """
    if not mmad_dir.exists():
        logger.error(f"MMAD directory not found: {mmad_dir}")
        return []
        
    # Load the metadata JSON
    mmad_json_path = mmad_dir / "mmad.json"
    if not mmad_json_path.exists():
        # Try case sensitive fallback
        mmad_json_path = mmad_dir / "MMAD.json"
        
    mmad_data = {}
    if mmad_json_path.exists():
        logger.info(f"Loading metadata from {mmad_json_path}")
        with open(mmad_json_path, 'r', encoding='utf-8') as f:
            mmad_data = json.load(f)
    else:
        logger.warning("mmad.json not found, metadata might be incomplete.")
    
    # Load VisA QA data
    visa_qa_data = load_visa_qa_data(mmad_dir)

    traces = []
    
    # Scan for .txt files
    logger.info("Scanning for text descriptions...")
    txt_files = list(mmad_dir.rglob("*.txt"))
    logger.info(f"Found {len(txt_files)} text files.")
    
    for txt_file in txt_files:
        # Find corresponding image (assumed same stem, png/jpg)
        # We don't strictly need the image to process the text, but we need the ID/Path
        
        # Construct relative path to match JSON keys
        try:
            rel_path = txt_file.relative_to(mmad_dir)
            # JSON keys usually point to the image .png, not .txt
            # Try replacing extension
            rel_path_str = str(rel_path)
            
            # Common image extensions to check (both lowercase and uppercase)
            img_key = None
            for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
                candidate = rel_path.with_suffix(ext)
                candidate_str = str(candidate).replace("\\", "/")
                if candidate_str in mmad_data:
                    img_key = candidate_str
                    break
            
            # If still not found, try case-insensitive search
            if not img_key:
                rel_path_lower = str(rel_path.with_suffix('')).replace("\\", "/").lower()
                for key in mmad_data.keys():
                    key_without_ext = key.rsplit('.', 1)[0].lower()
                    if key_without_ext == rel_path_lower:
                        img_key = key
                        break
            
            # Special case: MVTec-AD txt files -> DS-MVTec keys in mmad.json
            # MVTec-AD/bottle/test/broken_large/000.txt -> DS-MVTec/bottle/image/broken_large/000.png
            if not img_key and str(rel_path).startswith("MVTec-AD"):
                # Transform path: MVTec-AD/{product}/test/{defect}/{file} -> DS-MVTec/{product}/image/{defect}/{file}
                rel_str = str(rel_path).replace("\\", "/")
                parts = rel_str.split("/")
                if len(parts) >= 5:  # MVTec-AD/product/test/defect/file.txt
                    product = parts[1]
                    defect = parts[3]
                    filename = parts[4]
                    # Try DS-MVTec format
                    for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
                        ds_key = f"DS-MVTec/{product}/image/{defect}/{filename.rsplit('.', 1)[0]}{ext}"
                        if ds_key in mmad_data:
                            img_key = ds_key
                            logger.debug(f"Mapped MVTec-AD -> DS-MVTec: {rel_str} -> {ds_key}")
                            break
            
            if not img_key:
                # If exact match failed, try to infer from path components
                # e.g. DS-MVTec/bottle/image/broken_large/000.png
                pass 
            
            # If we still have no key, we can't get precise metadata (Location/Type)
            # But we can infer Answer/Type from folder names as fallback
            metadata = None
            if img_key:
                metadata = get_metadata_from_json(mmad_data, img_key, visa_qa_data)
            
            # Also check if this txt_file path directly matches VisA QA data
            if not metadata:
                rel_path_str = str(rel_path).replace("\\", "/")
                metadata = get_metadata_from_json(mmad_data, rel_path_str, visa_qa_data)
            
            # For MVTec-AD: if we have metadata but no location, try mask analysis
            if metadata and metadata.get("location") is None and "MVTec-AD" in str(txt_file):
                try:
                    rel_parts = list(txt_file.relative_to(mmad_dir).parts)
                    if len(rel_parts) >= 5 and metadata.get("answer") == "Yes":
                        product = rel_parts[1]
                        defect = rel_parts[3]
                        filename = rel_parts[4].rsplit('.', 1)[0]
                        mask_path = mmad_dir / "MVTec-AD" / product / "ground_truth" / defect / f"{filename}_mask.png"
                        if mask_path.exists():
                            location = get_location_from_mask(mask_path)
                            if location:
                                metadata["location"] = location
                                logger.debug(f"Added location from mask for {txt_file}: {location}")
                except Exception as e:
                    logger.warning(f"Error adding mask location for {txt_file}: {e}")
            
            if not metadata:
                # Fallback logic - this means we couldn't find the entry in mmad.json or VisA QA
                parts = txt_file.parts
                metadata = {"answer": "No", "type": None, "location": None}
                
                logger.debug(f"No mmad.json/VisA QA match for {txt_file}, using fallback")
                
                if "good" in parts or "Good" in parts or "train" in parts:
                    metadata["answer"] = "No"
                else:
                    metadata["answer"] = "Yes"
                    # Try to guess type from folder (parent folder often is defect type)
                    if len(parts) > 2:
                        defect_folder = parts[-2]
                        # Don't use generic folder names as type
                        if defect_folder.lower() not in ['bad', 'test', 'image', 'images']:
                            metadata["type"] = defect_folder.replace('_', ' ').title()
                        # Don't set "bad" as fallback - leave as None if we can't determine
                    
                    # For MVTec-AD: try to analyze mask for location
                    # MVTec-AD/{product}/test/{defect}/{file}.txt -> ground_truth/{defect}/{file}_mask.png
                    if "MVTec-AD" in str(txt_file):
                        try:
                            # Construct mask path
                            # e.g. MVTec-AD/bottle/test/broken_large/000.txt -> MVTec-AD/bottle/ground_truth/broken_large/000_mask.png
                            rel_parts = list(txt_file.relative_to(mmad_dir).parts)
                            if len(rel_parts) >= 5:
                                product = rel_parts[1]  # bottle
                                defect = rel_parts[3]   # broken_large
                                filename = rel_parts[4].rsplit('.', 1)[0]  # 000
                                
                                mask_path = mmad_dir / "MVTec-AD" / product / "ground_truth" / defect / f"{filename}_mask.png"
                                
                                if mask_path.exists():
                                    location = get_location_from_mask(mask_path)
                                    if location:
                                        metadata["location"] = location
                                        logger.debug(f"Got location from mask: {location}")
                        except Exception as e:
                            logger.warning(f"Error getting mask location for {txt_file}: {e}")
            
            # Read text content
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read()
            except Exception as e:
                logger.warning(f"Failed to read {txt_file}: {e}")
                continue
                
            if not content.strip():
                continue

            # Create trace
            image_id = txt_file.stem # Simple ID
            # Or use full path ID to avoid collisions across categories
            safe_id = str(rel_path.with_suffix('')).replace("\\", "_").replace("/", "_")
            
            trace = format_trace(safe_id, content, metadata)
            traces.append(trace)
            
            # Save to disk if requested
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                out_path = output_dir / f"trace_{safe_id}.json"
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(trace, f, indent=2)
                    
        except Exception as e:
            logger.error(f"Error processing {txt_file}: {e}")
            continue

    logger.info(f"Successfully processed {len(traces)} traces.")
    return traces

def load_certain_objects(category: str, mmad_dir: Path = MMAD_DIR) -> List[Dict]:
    """
    Load traces only for a specific category (e.g., 'bottle', 'screw').
    """
    all_traces = load_everything(mmad_dir, output_dir=None)
    filtered = [t for t in all_traces if category in t['image_id'] or category in t['reasoning']]
    return filtered

if __name__ == "__main__":
    # When run as script, perform the full conversion to disk
    print("Starting full dataset conversion...")
    load_everything(output_dir=OUTPUT_DIR)
    print(f"Done. Output saved to {OUTPUT_DIR}")



