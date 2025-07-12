"""
Configuration file for UI Element Detection Pipeline
Contains model configurations and default settings
"""

# Flash Model Configuration
# Used for center point detection from bounding box images
FLASH_MODEL_CONFIG = {
    "model_id": "gemini-2.5-flash",
    "temperature": 0.0,
    "thinking_budget": 0
}

# Detect Best Box Model Configuration  
# Used for context-aware best box selection using both images
DETECT_BEST_BOX_CONFIG = {
    "model_id": "gemini-2.5-pro",
    "temperature": 0.0,
    "thinking_budget": 1000
}

# Default Settings
DEFAULT_MONITOR_ID = 1
DEFAULT_TARGET_ITEM = "cat with laptop in the background"