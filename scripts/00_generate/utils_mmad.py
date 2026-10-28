"""
Utility functions for MMAD trace generation
"""
import re
import json
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

def sanitize_filename(text: str) -> str:
    """Sanitize text for use in filenames"""
    text = re.sub(r'[<>:"/\\|?*]', '_', text)
    text = text.strip()
    return text[:200]  # Limit length

def load_mmad_dataset(mmad_dir: Path) -> Dict[str, Any]:
    """
    Load MMAD dataset structure
    
    Returns:
        Dict with structure:
        {
            'images': [...],
            'questions': [...],
            'domain_knowledge': {...}
        }
    """
    dataset = {
        'images': [],
        'questions': [],
        'domain_knowledge': {}
    }
    
    # Load domain knowledge if exists
    domain_knowledge_file = mmad_dir / "domain_knowledge.json"
    if domain_knowledge_file.exists():
        with open(domain_knowledge_file, 'r', encoding='utf-8') as f:
            dataset['domain_knowledge'] = json.load(f)
        logger.info(f"Loaded domain knowledge from {domain_knowledge_file}")
    
    # TODO: Load actual MMAD questions/images
    # This depends on your MMAD dataset structure
    
    return dataset

def parse_json_response(response_text: str) -> Dict:
    """
    Parse JSON from Gemini response, handling markdown code blocks
    """
    # Try to extract JSON from markdown code blocks
    json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response_text, re.IGNORECASE)
    if json_match:
        json_str = json_match.group(1)
    else:
        json_str = response_text
    
    # Parse JSON
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {e}")
        logger.debug(f"Response text: {response_text[:500]}")
        raise

def validate_trace_structure(trace: Dict, expected_image_id: str) -> bool:
    """
    Validate that trace has correct structure
    
    Required fields:
    - image_id
    - reasoning (with <think>, <answer> tags)
    - location (if anomalous)
    - type (if anomalous)
    """
    if 'image_id' not in trace:
        logger.warning(f"Missing image_id in trace")
        return False
    
    if trace['image_id'] != expected_image_id:
        logger.warning(f"Image ID mismatch: expected {expected_image_id}, got {trace['image_id']}")
        return False
    
    if 'reasoning' not in trace:
        logger.warning(f"Missing reasoning field for {expected_image_id}")
        return False
    
    reasoning = trace['reasoning']
    
    # Check for required XML tags
    if '<think>' not in reasoning or '</think>' not in reasoning:
        logger.warning(f"Missing <think> tags in reasoning for {expected_image_id}")
        return False
    
    if '<answer>' not in reasoning or '</answer>' not in reasoning:
        logger.warning(f"Missing <answer> tags in reasoning for {expected_image_id}")
        return False
    
    # Extract answer
    answer_match = re.search(r'<answer>(Yes|No)</answer>', reasoning)
    if not answer_match:
        logger.warning(f"Invalid <answer> format for {expected_image_id}")
        return False
    
    answer = answer_match.group(1)
    
    # For anomalous (Yes), check for location and type
    if answer == "Yes":
        if '<location>' not in reasoning or '</location>' not in reasoning:
            logger.warning(f"Missing <location> tags for anomalous sample {expected_image_id}")
            return False
        
        if '<type>' not in reasoning or '</type>' not in reasoning:
            logger.warning(f"Missing <type> tags for anomalous sample {expected_image_id}")
            return False
    
    return True
