"""
Configuration file for UI Element Detection Pipeline
Contains model configurations and default settings
"""

# Detect Best Box Model Configuration  
# Used for context-aware best box selection using both images
DETECT_BEST_BOX_CONFIG = {
    "model_id": "gemini-2.5-pro",
    "temperature": 0.0,
    "thinking_budget": 10000
}

# Precision Point Model Configuration
# Used for precise point detection within cropped regions
PRECISION_POINT_CONFIG = {
    "model_id": "gemini-2.5-flash",
    "temperature": 0.0,
    "thinking_budget": 0,
    "min_confidence": 10
}

# Detector Model Configuration
# Used for bounding box detection in octants
DETECTOR_MODEL_CONFIG = {
    "model_id": "gemini-2.5-flash",
    "temperature": 0.0,
    "thinking_budget": 0,
    "min_confidence": 30,
    "max_detections_per_octant": 6,
    "default_overlap_percentage": 0.25,
    "nms_iou_threshold": 0.5
}

# Default Settings
DEFAULT_MONITOR_ID = 1
DEFAULT_TARGET_ITEM = "cat with laptop in the background"