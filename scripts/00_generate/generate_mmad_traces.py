"""
Generate reasoning traces for MMAD dataset using Gemini batch processing
"""
import json
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image
from tqdm import tqdm
from pydantic import BaseModel, Field, ValidationError
from google import genai
from google.genai.types import Content, Part, GenerateContentConfig
import config_mmad as config
import utils_mmad as utils

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Pydantic models for validation
class ReasoningTrace(BaseModel):
    image_id: str = Field(..., description="Image identifier")
    reasoning: str = Field(..., description="Full reasoning with XML tags")

class BatchReasoningTraces(BaseModel):
    traces: List[ReasoningTrace] = Field(..., description="List of reasoning traces")

def load_system_prompt() -> str:
    """Load the system prompt for industrial inspection"""
    prompt_file = config.PROMPTS_DIR / "inspector_prompt.txt"
    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"System prompt file not found: {prompt_file}")
        sys.exit(1)

def load_visa_qa_data() -> Dict[str, Dict[str, str]]:
    """
    Load VisA QA.json files to extract proper defect types and locations.
    
    Returns:
        Dict mapping relative image path -> {'defect_type': str, 'location': str}
    """
    visa_qa_map = {}
    visa_dir = config.MMAD_DIR / "VisA"
    
    if not visa_dir.exists():
        logger.warning(f"VisA directory not found: {visa_dir}")
        return visa_qa_map
    
    logger.info("Loading VisA QA.json files...")
    
    for product_dir in visa_dir.iterdir():
        if not product_dir.is_dir():
            continue
        
        qa_file = product_dir / "QA.json"
        if not qa_file.exists():
            logger.warning(f"QA.json not found for {product_dir.name}")
            continue
        
        try:
            with open(qa_file, 'r', encoding='utf-8') as f:
                qa_data = json.load(f)
            
            for img_rel_path, qa_entry in qa_data.items():
                conversations = qa_entry.get('conversation', [])
                
                defect_type = "unknown"
                location = "unknown"
                
                for conv in conversations:
                    conv_type = conv.get('type', '')
                    answer = conv.get('Answer', '')
                    options = conv.get('Options', {})
                    
                    # Extract defect type
                    if conv_type == 'Defect Classification' and answer in options:
                        defect_type = options[answer]
                    
                    # Extract location
                    elif conv_type == 'Defect Localization' and answer in options:
                        location = options[answer]
                
                # Create full relative path: VisA/product/test/bad/000.JPG
                full_rel_path = f"VisA/{product_dir.name}/{img_rel_path}"
                visa_qa_map[full_rel_path] = {
                    'defect_type': defect_type,
                    'location': location
                }
        
        except Exception as e:
            logger.error(f"Error loading QA.json for {product_dir.name}: {e}")
    
    logger.info(f"Loaded QA data for {len(visa_qa_map)} VisA images")
    return visa_qa_map

def load_mmad_questions() -> List[Dict[str, Any]]:
    """
    Load MMAD images and their corresponding text descriptions.
    """
    mmad_questions = []
    
    logger.info(f"Scanning for images in {config.MMAD_DIR}...")
    
    # Load VisA QA data first
    visa_qa_map = load_visa_qa_data()
    
    # Walk through the directory to find images
    images_dir = config.MMAD_DIR
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    count = 0
    for img_path in images_dir.glob("**/*"):
        if img_path.suffix.lower() in image_extensions:
            # Check for corresponding .txt file
            txt_path = img_path.with_suffix('.txt')
            
            existing_description = ""
            if txt_path.exists():
                try:
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        existing_description = f.read().strip()
                except Exception as e:
                    logger.warning(f"Could not read text file {txt_path}: {e}")
            
            # Determine metadata from path
            # Path structure usually: .../category/image/defect_type/filename
            # or .../dataset/category/image/defect_type/filename
            
            parts = img_path.parts
            category = "component"
            defect_type = "unknown"
            location = "unknown"
            ground_truth = "No"
            
            # Get relative path for VisA lookup
            rel_path = str(img_path.relative_to(config.MMAD_DIR))
            
            # Check if this is a VisA image with QA data
            if rel_path in visa_qa_map:
                qa_info = visa_qa_map[rel_path]
                defect_type = qa_info['defect_type']
                location = qa_info['location']
                ground_truth = "Yes"  # VisA QA only contains anomalies
                # Extract category from path
                category = parts[parts.index('VisA') + 1] if 'VisA' in parts else "component"
            else:
                # Original heuristic logic for non-VisA or VisA good images
                if "good" in parts or "Good" in parts:
                    ground_truth = "No"
                    defect_type = "None"
                    location = "N/A"
                else:
                    ground_truth = "Yes"
                    # Try to guess defect type from parent folder (for MVTec, etc.)
                    if len(parts) > 1:
                        folder_name = parts[-2]
                        # Only use folder name if it's not generic "bad"
                        if folder_name.lower() != "bad":
                            defect_type = folder_name
                        # else keep as "unknown"
                
                # Try to find category (usually folder before 'image' or parent of parent)
                # Adjust this based on your specific folder structure if needed
                if "bottle" in parts: category = "bottle"
                elif "carpet" in parts: category = "carpet"
                elif "cable" in parts: category = "cable"
                elif "capsule" in parts: category = "capsule"
                elif "hazelnut" in parts: category = "hazelnut"
                elif "metal_nut" in parts: category = "metal_nut"
                elif "pill" in parts: category = "pill"
                elif "screw" in parts: category = "screw"
                elif "toothbrush" in parts: category = "toothbrush"
                elif "transistor" in parts: category = "transistor"
                elif "zipper" in parts: category = "zipper"
            
            image_id = utils.sanitize_filename(str(img_path.relative_to(config.MMAD_DIR)))

            mmad_questions.append({
                'image_id': image_id,
                'image_path': img_path,
                'question': 'Is there any defect in this image?',
                'ground_truth': ground_truth,
                'defect_type': defect_type,
                'location': location,
                'category': category,
                'existing_description': existing_description
            })
            count += 1
            
            if count % 1000 == 0:
                logger.info(f"Found {count} images so far...")

    logger.info(f"Loaded {len(mmad_questions)} MMAD images")
    return mmad_questions

def create_batch(
    questions: List[Dict],
    processed_ids: set,
    batch_size: int,
    attempt_counts: Dict,
    max_attempts: int
) -> List[Dict]:
    """Create a batch of images to process"""
    batch = []
    
    for q in questions:
        if len(batch) >= batch_size:
            break
        
        image_id = q['image_id']
        if image_id in processed_ids:
            continue
        
        attempts = attempt_counts.get(image_id, 0)
        if attempts >= max_attempts:
            continue
        
        if not q['image_path'].exists():
            logger.warning(f"Image not found: {q['image_path']}")
            continue
        
        batch.append(q)
        attempt_counts[image_id] = attempts + 1
    
    return batch

def format_batch_prompt(batch: List[Dict], system_prompt: str) -> str:
    """Format batch prompt for multiple images"""
    prompt_parts = []
    
    prompt_parts.append(f"You will receive {len(batch)} images to inspect.")
    prompt_parts.append("For EACH image, generate a complete inspection report following the specified format.")
    prompt_parts.append("")
    
    for idx, item in enumerate(batch, 1):
        prompt_parts.append(f"--- IMAGE {idx} ---")
        prompt_parts.append(f"Image ID: {item['image_id']}")
        prompt_parts.append(f"Product Type: {item.get('category', 'component')}")
        
        # Add ground truth hints (helps Gemini generate correct traces)
        if item.get('ground_truth') == 'Yes':
            prompt_parts.append(f"Expected: ANOMALOUS")
            if 'defect_type' in item and item['defect_type'] not in ['unknown', 'bad']:
                prompt_parts.append(f"Defect Type Hint: {item['defect_type']}")
            if 'location' in item and item['location'] not in ['unknown', 'Unknown']:
                prompt_parts.append(f"Defect Location Hint: {item['location']}")
        else:
            prompt_parts.append(f"Expected: NORMAL")
        
        # Provide the existing description as context to help the model
        if item.get('existing_description'):
            prompt_parts.append(f"Reference Observation:\n{item['existing_description']}")
            prompt_parts.append("IMPORTANT: Use the details in the 'Reference Observation' to write your <think> block, but strictly follow the required output format and structure.")
        
        prompt_parts.append("")
    
    prompt_parts.append("Return a JSON array with traces for all images.")
    
    return "\n".join(prompt_parts)

def process_batch(
    batch: List[Dict],
    client: genai.Client,
    model: str,
    system_prompt: str
) -> Dict[str, str]:
    """
    Process a batch of images and return traces
    
    Returns:
        Dict mapping image_id -> reasoning trace
    """
    batch_id = str(uuid.uuid4())[:8]
    logger.info(f"Processing batch {batch_id} with {len(batch)} images")
    
    # Prepare images and text prompt
    content_parts = []
    
    # Add text prompt
    text_prompt = format_batch_prompt(batch, system_prompt)
    content_parts.append(Part.from_text(text=text_prompt))
    
    # Add images
    for item in batch:
        img = Image.open(item['image_path'])
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        # Gemini expects PIL images directly
        content_parts.append(Part.from_image(img))
    
    # Generate
    generate_config = GenerateContentConfig(
        response_mime_type="application/json",
        system_instruction=[Part.from_text(text=system_prompt)]
    )
    
    full_response = ""
    stream = client.models.generate_content_stream(
        model=model,
        contents=[Content(role="user", parts=content_parts)],
        config=generate_config
    )
    
    for chunk in stream:
        if chunk.text:
            full_response += chunk.text
    
    # Parse response
    response_dict = utils.parse_json_response(full_response)
    
    # Handle different response formats
    if isinstance(response_dict, list):
        response_dict = {'traces': response_dict}
    
    if 'traces' not in response_dict and 'image_id' in response_dict:
        response_dict = {'traces': [response_dict]}
    
    # Validate
    validated = BatchReasoningTraces(**response_dict)
    
    # Create mapping
    traces = {}
    for trace in validated.traces:
        traces[trace.image_id] = trace.reasoning
    
    return traces

def run():
    """Main execution function"""
    logger.info("=== Starting MMAD Reasoning Trace Generation ===")
    
    # Load system prompt
    system_prompt = load_system_prompt()
    logger.info("Loaded system prompt")
    
    # Load MMAD questions
    questions = load_mmad_questions()
    if not questions:
        logger.error("No MMAD questions loaded!")
        sys.exit(1)
    
    # Check which traces already exist
    processed_ids = set()
    logger.info("Scanning existing traces...")
    for trace_file in config.REASONING_TRACES_DIR.glob("trace_*.json"):
        try:
            with open(trace_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if 'image_id' in data:
                    processed_ids.add(data['image_id'])
        except:
            pass
    
    logger.info(f"Already processed: {len(processed_ids)}")
    logger.info(f"Remaining: {len(questions) - len(processed_ids)}")
    
    # Initialize API
    api_keys = config.GEMINI_API_KEYS
    current_key_idx = 0
    client = genai.Client(api_key=api_keys[current_key_idx])
    
    attempt_counts = {}
    permanently_skipped = set()
    
    # Process in batches
    with tqdm(total=len(questions), desc="Generating traces", initial=len(processed_ids)) as pbar:
        while True:
            # Get remaining questions
            remaining = [
                q for q in questions
                if q['image_id'] not in processed_ids
                and q['image_id'] not in permanently_skipped
            ]
            
            if not remaining:
                break
            
            # Check for max retries
            for q in remaining:
                if attempt_counts.get(q['image_id'], 0) >= config.MAX_RETRIES:
                    if q['image_id'] not in permanently_skipped:
                        logger.warning(f"Permanently skipping {q['image_id']} after {config.MAX_RETRIES} attempts")
                        permanently_skipped.add(q['image_id'])
            
            # Create batch
            batch = create_batch(
                remaining,
                processed_ids,
                config.BATCH_SIZE,
                attempt_counts,
                config.MAX_RETRIES
            )
            
            if not batch:
                break
            
            # Process batch
            success = False
            retry_count = 0
            
            while not success and retry_count < len(api_keys):
                try:
                    traces = process_batch(
                        batch,
                        client,
                        config.GEMINI_MODEL,
                        system_prompt
                    )
                    
                    # Validate and save
                    saved = 0
                    for item in batch:
                        image_id = item['image_id']
                        
                        if image_id in traces:
                            # Validate structure
                            trace_data = {
                                'image_id': image_id,
                                'reasoning': traces[image_id]
                            }
                            
                            if utils.validate_trace_structure(trace_data, image_id):
                                # Save trace
                                sanitized = utils.sanitize_filename(image_id)
                                trace_file = config.REASONING_TRACES_DIR / f"trace_{sanitized}.json"
                                
                                with open(trace_file, 'w', encoding='utf-8') as f:
                                    json.dump(trace_data, f, indent=2, ensure_ascii=False)
                                
                                processed_ids.add(image_id)
                                saved += 1
                                logger.debug(f"✅ Saved trace for {image_id}")
                            else:
                                logger.warning(f"⚠️  Invalid trace structure for {image_id}")
                        else:
                            logger.warning(f"⚠️  No trace returned for {image_id}")
                    
                    pbar.update(saved)
                    logger.info(f"✅ Saved {saved}/{len(batch)} traces")
                    
                    success = True
                    
                except Exception as e:
                    error_str = str(e)
                    logger.error(f"Error processing batch: {error_str}")
                    
                    if '429' in error_str or 'RESOURCE_EXHAUSTED' in error_str:
                        logger.warning(f"Rate limit hit, waiting {config.RATE_LIMIT_WAIT}s...")
                        time.sleep(config.RATE_LIMIT_WAIT)
                    
                    # Rotate API key
                    current_key_idx = (current_key_idx + 1) % len(api_keys)
                    logger.info(f"Switching to API key {current_key_idx + 1}/{len(api_keys)}")
                    client = genai.Client(api_key=api_keys[current_key_idx])
                    retry_count += 1
    
    # Summary
    logger.info("")
    logger.info("=== Generation Complete ===")
    logger.info(f"Successfully generated: {len(processed_ids)}")
    logger.info(f"Total traces: {len(list(config.REASONING_TRACES_DIR.glob('trace_*.json')))}")
    logger.info(f"Permanently skipped: {len(permanently_skipped)}")
    
    if permanently_skipped:
        logger.warning(f"Failed IDs: {list(permanently_skipped)[:10]}")

if __name__ == "__main__":
    run()
