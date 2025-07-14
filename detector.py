#!/usr/bin/env python3
"""
Octant Bounding Box Detection with Overlap Support
Improved accuracy through better prompts, NMS, and validation.
"""

import os
import json
import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from google import genai
from google.genai import types
from PIL import Image, ImageDraw, ImageFont
from capture import capture_overlapping_octants, map_overlapping_coordinates
from divider import divide_image_into_overlapping_octants
import ctypes
from typing import List, Dict, Tuple, Optional

# Set DPI awareness for high-resolution capture
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except:
        pass

# --- Configuration ---
# Import configuration from config.py
from config import DETECTOR_MODEL_CONFIG

# Gemini Client Initialization
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))

# Model Configuration
MODEL_ID = DETECTOR_MODEL_CONFIG["model_id"]
TEMPERATURE = DETECTOR_MODEL_CONFIG["temperature"]
THINKING_BUDGET = DETECTOR_MODEL_CONFIG["thinking_budget"]

# Detection Configuration
MIN_CONFIDENCE = DETECTOR_MODEL_CONFIG["min_confidence"]
MAX_DETECTIONS_PER_OCTANT = DETECTOR_MODEL_CONFIG["max_detections_per_octant"]
DEFAULT_OVERLAP_PERCENTAGE = DETECTOR_MODEL_CONFIG["default_overlap_percentage"]
NMS_IOU_THRESHOLD = DETECTOR_MODEL_CONFIG["nms_iou_threshold"]

# Folder configuration
SCREENSHOTS_FOLDER = "screenshots"
DETECTIONS_FOLDER = "detections"
ANNOTATED_FOLDER = os.path.join(DETECTIONS_FOLDER, "annotated")
RESULTS_FOLDER = os.path.join(DETECTIONS_FOLDER, "results")
CROPPED_FOLDER = os.path.join(DETECTIONS_FOLDER, "cropped")


def calculate_iou(box1: Dict, box2: Dict) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1, box2: Dictionaries with 'xmin', 'ymin', 'xmax', 'ymax' keys
        
    Returns:
        IoU value between 0 and 1
    """
    # Calculate intersection
    x_left = max(box1['xmin'], box2['xmin'])
    y_top = max(box1['ymin'], box2['ymin'])
    x_right = min(box1['xmax'], box2['xmax'])
    y_bottom = min(box1['ymax'], box2['ymax'])
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union
    box1_area = (box1['xmax'] - box1['xmin']) * (box1['ymax'] - box1['ymin'])
    box2_area = (box2['xmax'] - box2['xmin']) * (box2['ymax'] - box2['ymin'])
    union_area = box1_area + box2_area - intersection_area
    
    return intersection_area / union_area if union_area > 0 else 0.0


def non_maximum_suppression(detections: List[Dict], 
                           iou_threshold: float = NMS_IOU_THRESHOLD,
                           confidence_threshold: float = MIN_CONFIDENCE) -> List[Dict]:
    """
    Apply Non-Maximum Suppression to remove overlapping detections.
    
    Args:
        detections: List of detections with confidence scores
        iou_threshold: IoU threshold for suppression
        confidence_threshold: Minimum confidence to keep detection
        
    Returns:
        List of filtered detections
    """
    if not detections:
        return []
    
    # Extract confidence scores
    def get_confidence(detection):
        label = detection.get('label', '')
        match = re.search(r'\((\d+)%\)', label)
        return int(match.group(1)) if match else 0
    
    # Filter by confidence threshold
    filtered = [d for d in detections if get_confidence(d) >= confidence_threshold]
    
    if not filtered:
        return []
    
    # Sort by confidence (descending)
    filtered.sort(key=get_confidence, reverse=True)
    
    # Apply NMS
    keep = []
    while filtered:
        # Take the detection with highest confidence
        best = filtered.pop(0)
        keep.append(best)
        
        # Remove all detections with high IoU
        filtered = [d for d in filtered if calculate_iou(best, d) < iou_threshold]
    
    return keep


def validate_detection(detection: Dict, image_width: int, image_height: int) -> bool:
    """
    Validate if a detection is reasonable.
    
    Args:
        detection: Detection dictionary
        image_width: Width of the image
        image_height: Height of the image
        
    Returns:
        True if detection is valid
    """
    # Check if detection is too small
    min_size = 10  # pixels
    width = detection['xmax'] - detection['xmin']
    height = detection['ymax'] - detection['ymin']
    
    if width < min_size or height < min_size:
        return False
    
    # Check if detection is too large (likely false positive)
    if width > image_width * 0.9 or height > image_height * 0.9:
        return False
    
    # Check aspect ratio for reasonable UI elements
    aspect_ratio = width / height if height > 0 else 0
    if aspect_ratio > 20 or aspect_ratio < 0.05:  # Too wide or too tall
        return False
    
    # Check if detection is at image border (often cut-off elements)
    border_threshold = 5
    if (detection['xmin'] < border_threshold or 
        detection['ymin'] < border_threshold or
        detection['xmax'] > image_width - border_threshold or
        detection['ymax'] > image_height - border_threshold):
        # Only reject if it's touching multiple borders (likely artifact)
        borders_touched = sum([
            detection['xmin'] < border_threshold,
            detection['ymin'] < border_threshold,
            detection['xmax'] > image_width - border_threshold,
            detection['ymax'] > image_height - border_threshold
        ])
        if borders_touched >= 2:
            return False
    
    return True


def merge_overlapping_detections(detections: List[Dict], 
                               iou_threshold: float = NMS_IOU_THRESHOLD) -> List[Dict]:
    """
    Merge overlapping detections from different octants.
    
    Args:
        detections: List of all detections from all octants
        iou_threshold: Minimum IoU to consider boxes as duplicates
        
    Returns:
        List of merged detections
    """
    if not detections:
        return []
    
    # Sort by confidence (assuming confidence is in the label)
    def get_confidence(detection):
        label = detection.get('label', '')
        match = re.search(r'\((\d+)%\)', label)
        return int(match.group(1)) if match else 0
    
    sorted_detections = sorted(detections, key=get_confidence, reverse=True)
    
    merged = []
    used = set()
    
    for i, det1 in enumerate(sorted_detections):
        if i in used:
            continue
            
        # Find all overlapping detections
        overlapping = [det1]
        for j, det2 in enumerate(sorted_detections[i+1:], i+1):
            if j not in used and calculate_iou(det1, det2) >= iou_threshold:
                overlapping.append(det2)
                used.add(j)
        
        # Merge overlapping boxes (weighted average by confidence)
        if len(overlapping) == 1:
            merged.append(det1)
        else:
            # Calculate weighted average box
            total_conf = sum(get_confidence(d) for d in overlapping)
            
            if total_conf > 0:
                avg_box = {
                    'xmin': sum(d['xmin'] * get_confidence(d) for d in overlapping) / total_conf,
                    'ymin': sum(d['ymin'] * get_confidence(d) for d in overlapping) / total_conf,
                    'xmax': sum(d['xmax'] * get_confidence(d) for d in overlapping) / total_conf,
                    'ymax': sum(d['ymax'] * get_confidence(d) for d in overlapping) / total_conf,
                }
            else:
                # Simple average if no confidence
                avg_box = {
                    'xmin': sum(d['xmin'] for d in overlapping) / len(overlapping),
                    'ymin': sum(d['ymin'] for d in overlapping) / len(overlapping),
                    'xmax': sum(d['xmax'] for d in overlapping) / len(overlapping),
                    'ymax': sum(d['ymax'] for d in overlapping) / len(overlapping),
                }
            
            # Use the label from the highest confidence detection
            avg_box['label'] = det1['label']
            avg_box['octants'] = [d.get('octant', -1) for d in overlapping]
            avg_box['merged_count'] = len(overlapping)
            
            # Round coordinates to integers
            for key in ['xmin', 'ymin', 'xmax', 'ymax']:
                avg_box[key] = int(round(avg_box[key]))
            
            merged.append(avg_box)
    
    return merged


def generate_context_aware_prompt(target: str) -> str:
    """
    Generate a context-aware prompt for better detection.
    
    Args:
        target: What to detect
        
    Returns:
        Prompt string
    """
    base_prompt = f"""You are an expert UI element detector analyzing a zoomed-in screenshot section.

STEP-BY-STEP REASONING (think internally, do not include in output):
1. Interpret the query "{target}": Generate possible synonyms, visual descriptions, or related terms. 
   - For example, "Brave icon", it might refer to the Brave web browser icon (lion logo), a browser shortcut, or an app launcher icon for Brave.
   - "battery icon", it might mean a battery status symbol in the taskbar, power indicator, or laptop charge icon.
   -"browser tab", it might include all visible tabs in a browser window, such as active/inactive tabs with titles or close buttons.
2. Consider multiples: If the target implies similar elements (e.g., all browser tabs), plan to detect all relevant instances.
3. Consider context: If the target is a small detail within a larger element (e.g., an ear in a photo), plan to detect both as separate items.
4. Focus on relevance: Ensure all detections directly match your interpretations; ignore unrelated elements.
5. Use these interpretations to guide your detection below. Prioritize the most likely matches based on visual similarity, interactivity, and common UI patterns.

YOUR TASK: Detect {target} in this image, using your interpreted descriptions from reasoning.

DETECTION GUIDELINES:
1. Detect ONLY elements that directly match or are highly relevant to your interpreted target. Do not include unrelated UI elements, even if they are prominent.
2. If multiple similar elements match the target (e.g., all visible browser tabs for 'browser tab'), include them all, prioritizing the most confident/relevant ones up to the limit.
3. Include elements that are at least 60% visible.
4. Prioritize elements that users can click or interact with (buttons, links, icons, menus).
5. Each element should have a clear, tight bounding box.
6. **CONTEXTUAL DETECTION**: If the target is a small detail within a larger, distinct element (e.g., a face within an advertisement, an icon within a button), you MUST return **two separate detections**: one for the specific target and another for the larger containing element.
7. For icons, consider recognizable logos, symbols, or shapes (e.g., a lion for Brave browser).
8. 
EXAMPLES OF WHAT TO DETECT:
- Interactive buttons, icons, and toolbar actions (e.g., save or print icons)
- Clickable links, hyperlinks, menus, and navigation elements
- Form controls (e.g., checkboxes, radio buttons, dropdowns)
- Application icons/shortcuts (e.g., browser icons like Chrome globe or Brave lion) and window controls (e.g., close, minimize)
- Tab headers and interactive section titles
- All related instances if multiple match (e.g., all visible browser tabs for 'browser tab');



OUTPUT FORMAT:
[{{"box_2d": [ymin, xmin, ymax, xmax], "label": "<precise description> (<confidence>%)"}}]

Where:
- box_2d: [ymin, xmin, ymax, xmax] normalized to 0-1000
- label: Specific description of the target + confidence ({MIN_CONFIDENCE}-100%)
- Include up to {MAX_DETECTIONS_PER_OCTANT} most relevant detections, with minimum of 2 detections.
- Only include detections with confidence >= {MIN_CONFIDENCE}%

EXAMPLE GOOD RESPONSES:
[{{"box_2d": [100, 200, 150, 280], "label": "File menu button (92%)"}}]
[{{"box_2d": [300, 400, 340, 480], "label": "Save document icon (88%)"}}, 
 {{"box_2d": [300, 500, 340, 580], "label": "Print icon (85%)"}}]
# Example: Detecting all browser tabs for target "browser tab"
[{{"box_2d": [100, 100, 150, 300], "label": "Browser tab titled 'Google' (95%)"}}, 
 {{"box_2d": [100, 310, 150, 510], "label": "Browser tab titled 'YouTube' (92%)"}}]
# Example: Detecting "Putin's ear" and also returning the ad it belongs to as a separate detection.
[{{"box_2d": [526, 626, 547, 646], "label": "Putin's ear (95%)"}}, 
 {{"box_2d": [444, 445, 675, 776], "label": "Whole image of ad with Putin (90%)"}}]

RETURN [] if no matching elements found with sufficient confidence."""

    return base_prompt


def process_single_octant_bbox(image_path: str, 
                             octant_id: int,
                             target: str = "target", 
                             verbose: bool = True) -> List[Dict]:
    """
    Process a single octant image to detect bounding boxes.
    
    Args:
        image_path: Path to the octant image
        octant_id: ID of the octant (0-7)
        target: What to look for in the image
        verbose: Whether to print status messages
        
    Returns:
        List of detected targets with bounding boxes
    """
    try:
        img = Image.open(image_path)
        if verbose:
            print(f"  Octant {octant_id} - Image size: {img.size[0]}x{img.size[1]}")
    except Exception as e:
        if verbose:
            print(f"  Octant {octant_id} - Error loading image: {e}")
        return []
    
    # Generate context-aware prompt
    prompt = generate_context_aware_prompt(target)
    generation_config = types.GenerateContentConfig(
        temperature=TEMPERATURE,
        response_mime_type="application/json",
        thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET)
    )
    
    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=[img, prompt],
            config=generation_config
        )
        
        detected_targets = json.loads(response.text)
        
        if verbose and detected_targets:
            print(f"    Found {len(detected_targets)} targets in octant {octant_id}")
        
        # Add octant ID to each detection
        for item in detected_targets:
            item['octant'] = octant_id
        
        return detected_targets
        
    except Exception as e:
        if verbose:
            print(f"  Octant {octant_id} - Error during detection: {e}")
        return []



def detect_items_bbox(target: str = "target",
                     monitor_id: int = 2,
                     verbose: bool = True,
                     use_overlap: bool = True,
                     overlap_percentage: float = DEFAULT_OVERLAP_PERCENTAGE,
                     merge_duplicates: bool = True,
                     fallback_fullscreen: bool = True,
                     skip_capture: bool = False,
                     multi_scale: bool = False) -> Dict:
    """
    Detection with overlapping octants and duplicate merging.
    
    Args:
        target: What to look for
        monitor_id: Monitor to capture
        verbose: Whether to print status messages
        use_overlap: Whether to use overlapping octants
        overlap_percentage: How much octants should overlap
        merge_duplicates: Whether to merge duplicate detections
        fallback_fullscreen: Whether to try full screen detection if octants fail
        skip_capture: Whether to skip screenshot capture (use existing screenshot_full.png)
        multi_scale: Whether to use multi-scale detection
        
    Returns:
        Dictionary with detection results
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create necessary folders
    for folder in [SCREENSHOTS_FOLDER, DETECTIONS_FOLDER, ANNOTATED_FOLDER, 
                   RESULTS_FOLDER, CROPPED_FOLDER]:
        os.makedirs(folder, exist_ok=True)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Octant Bounding Box Detection")
        print(f"Target: {target}")
        print(f"Overlap: {overlap_percentage*100}%")
        print(f"Min Confidence: {MIN_CONFIDENCE}%")
        print(f"Max Detections/Octant: {MAX_DETECTIONS_PER_OCTANT}")
        print(f"Multi-scale: {multi_scale}")
        print(f"{'='*60}\n")
    
    # Handle screenshot capture or use existing
    if skip_capture:
        # Use existing screenshot
        full_screenshot_path = os.path.join(SCREENSHOTS_FOLDER, "screenshot_full.png")
        if not os.path.exists(full_screenshot_path):
            return {'success': False, 'error': 'No existing screenshot found at screenshots/screenshot_full.png'}

        # Get image resolution
        with Image.open(full_screenshot_path) as img:
            full_resolution = img.size

        if verbose:
            print(f"Using existing screenshot: {full_screenshot_path}")
            print(f"Resolution: {full_resolution[0]}x{full_resolution[1]}")


        # Divide the existing image into octants
        full_img = Image.open(full_screenshot_path)

        # Use the divide_image_into_overlapping_octants function
        octants_data = divide_image_into_overlapping_octants(
            full_img,
            overlap_percentage=overlap_percentage if use_overlap else 0.0,
            target_resolution=None,
            upscale=True,
            screenshots_folder=SCREENSHOTS_FOLDER,
            verbose=verbose
        )

        if not octants_data:
            return {'success': False, 'error': 'Failed to divide image into octants'}

        octant_files = octants_data['octant_files']
        mapping_file = octants_data['mapping_file']

        full_resolution = octants_data['full_resolution']

        capture_result = {
            'octant_files': octant_files,
            'mapping_file': mapping_file,
            'full_resolution': full_resolution
        }
    else:
        # Original capture logic
        capture_result = capture_overlapping_octants(
            monitor_id=monitor_id,
            overlap_percentage=overlap_percentage if use_overlap else 0.0,
            verbose=verbose
        )

        if not capture_result:
            return {'success': False, 'error': 'Failed to capture octants'}

        octant_files = capture_result['octant_files']
        mapping_file = capture_result['mapping_file']
        full_resolution = capture_result['full_resolution']
        
    
    # Process octants in parallel
    all_detections = []
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_octant = {
            executor.submit(process_single_octant_bbox, octant_file, idx, target, verbose): idx
            for idx, octant_file in enumerate(octant_files)
        }
        
        for future in as_completed(future_to_octant):
            octant_id = future_to_octant[future]
            try:
                detections = future.result()
                all_detections.extend(detections)
            except Exception as e:
                if verbose:
                    print(f"Octant {octant_id} processing failed: {e}")
    
    # Convert octant coordinates to screen coordinates
    screen_detections = []
    
    for detection in all_detections:
        if 'box_2d' not in detection:
            continue
            
        ymin, xmin, ymax, xmax = detection['box_2d']
        octant_id = detection.get('octant', 0)
        
        # Map corners to screen coordinates
        try:
            x1, y1 = map_overlapping_coordinates(octant_id, xmin, ymin, mapping_file)
            x2, y2 = map_overlapping_coordinates(octant_id, xmax, ymax, mapping_file)
            
            screen_detection = {
                'xmin': int(x1),
                'ymin': int(y1),
                'xmax': int(x2),
                'ymax': int(y2),
                'label': detection.get('label', 'Unknown'),
                'octant': octant_id,
                'confidence': re.search(r'\((\d+)%\)', detection.get('label', '')).group(1) if re.search(r'\((\d+)%\)', detection.get('label', '')) else '0'
            }
            
            screen_detections.append(screen_detection)
            
        except Exception as e:
            if verbose:
                print(f"Failed to map coordinates for detection in octant {octant_id}: {e}")
    
    # Apply Non-Maximum Suppression instead of simple merging
    if merge_duplicates and len(screen_detections) > 1:
        # First apply NMS
        nms_detections = non_maximum_suppression(screen_detections)
        
        # Then validate detections
        final_detections = []
        for det in nms_detections:
            if validate_detection(det, full_resolution[0], full_resolution[1]):
                final_detections.append(det)
        
        if verbose:
            print(f"\nNMS: {len(screen_detections)} -> {len(nms_detections)} detections")
            print(f"Validation: {len(nms_detections)} -> {len(final_detections)} detections")
    else:
        # Still validate even without merging
        final_detections = []
        for det in screen_detections:
            if validate_detection(det, full_resolution[0], full_resolution[1]):
                final_detections.append(det)
    
    # Multi-scale detection (optional)
    if multi_scale and not final_detections:
        if verbose:
            print("\nTrying multi-scale detection...")
        # TODO: Implement multi-scale detection
        # This would involve processing the image at different scales
        # and combining the results
    
    # Fallback to full screen detection if no results
    if fallback_fullscreen and not final_detections:
        if verbose:
            print("\nNo detections in octants, trying full screen detection...")
        
        full_img_path = os.path.join(SCREENSHOTS_FOLDER, "screenshot_full.png")
        full_detections = process_single_octant_bbox(full_img_path, -1, target, verbose)
        
        # Convert normalized coordinates to screen coordinates
        for detection in full_detections:
            if 'box_2d' in detection:
                ymin, xmin, ymax, xmax = detection['box_2d']
                final_detections.append({
                    'xmin': int(xmin * full_resolution[0] / 1000),
                    'ymin': int(ymin * full_resolution[1] / 1000),
                    'xmax': int(xmax * full_resolution[0] / 1000),
                    'ymax': int(ymax * full_resolution[1] / 1000),
                    'label': detection.get('label', 'Unknown'),
                    'octant': -1,  # Full screen
                    'confidence': re.search(r'\((\d+)%\)', detection.get('label', '')).group(1) if re.search(r'\((\d+)%\)', detection.get('label', '')) else '0'
                })
    
    # Add IDs to final detections
    for idx, detection in enumerate(final_detections):
        detection['id'] = idx + 1
    
    # Create visualizations
    annotated_path = create_annotated_image(final_detections, full_resolution, timestamp, verbose)
    cropped_canvas_path = create_cropped_images_canvas(final_detections, full_resolution, timestamp, verbose)
    numbered_annotated_path = create_numbered_annotated_screenshot(final_detections, full_resolution, timestamp, verbose)
    
    # Save results
    results = {
        'timestamp': timestamp,
        'target_items': target,
        'use_overlap': use_overlap,
        'overlap_percentage': overlap_percentage,
        'total_detections': len(final_detections),
        'bounding_boxes': final_detections,
        'full_resolution': full_resolution,
        'annotated_image': annotated_path,
        'cropped_canvas': cropped_canvas_path,
        'numbered_annotated_image': numbered_annotated_path,
    }
    
    results_file = os.path.join(RESULTS_FOLDER, f"detection_results_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Detection Complete")
        print(f"Total targets found: {len(final_detections)}")
        print(f"Results saved: {results_file}")
        print(f"{'='*60}")
    
    return {
        'success': True,
        'bounding_boxes': final_detections,
        'annotated_image_path': annotated_path,
        'cropped_images_canvas': cropped_canvas_path,
        'numbered_annotated_image': numbered_annotated_path,
        'results_file': results_file
    }


def create_annotated_image(detections: List[Dict], 
                          full_resolution: Tuple[int, int],
                          timestamp: str,
                          verbose: bool = True) -> str:
    """Create annotated image with bounding boxes."""
    full_img_path = os.path.join(SCREENSHOTS_FOLDER, "screenshot_full.png")
    img = Image.open(full_img_path)
    draw = ImageDraw.Draw(img)
    
    # Try to load a font
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'cyan', 'magenta']
    
    for idx, bbox in enumerate(detections):
        color = colors[idx % len(colors)]
        
        # Draw rectangle
        draw.rectangle(
            [bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']], 
            outline=color, 
            width=3
        )
        
        # Draw label background
        label = f"{bbox['id']}: {bbox['label']}"
        bbox_width = draw.textbbox((0, 0), label, font=font)
        text_width = bbox_width[2] - bbox_width[0]
        text_height = bbox_width[3] - bbox_width[1]
        
        draw.rectangle(
            [bbox['xmin'], bbox['ymin'] - text_height - 4, 
             bbox['xmin'] + text_width + 4, bbox['ymin']], 
            fill=color
        )
        
        # Draw label text
        draw.text(
            (bbox['xmin'] + 2, bbox['ymin'] - text_height - 2), 
            label, 
            fill='white', 
            font=font
        )
    
    # Save annotated image
    annotated_path = os.path.join(ANNOTATED_FOLDER, f"annotated_{timestamp}.png")
    img.save(annotated_path)
    
    if verbose:
        print(f"Annotated image saved: {annotated_path}")
    
    return annotated_path


def create_cropped_images_canvas(detections: List[Dict],
                               full_resolution: Tuple[int, int],
                               timestamp: str,
                               verbose: bool = True) -> Optional[str]:
    """Create a canvas with all cropped detections with numbered labels."""
    if not detections:
        return None
    
    full_img_path = os.path.join(SCREENSHOTS_FOLDER, "screenshot_full.png")
    full_img = Image.open(full_img_path)
    
    # Calculate canvas size with tighter spacing
    padding = 10  # Reduced from 20
    border_width = 3
    min_display_size = 80  # Minimum size for small icons to be visible
    max_crop_width = 200  # Reduced from 300
    max_crop_height = 200  # Reduced from 300
    label_height_estimate = 40  # Estimated height for larger labels
    label_spacing = 3  # Space between label and image
    
    # Create canvas
    crops_per_row = 6  # Increased from 4 to make canvas more compact
    rows_needed = (len(detections) + crops_per_row - 1) // crops_per_row
    
    canvas_width = (max_crop_width + padding) * min(len(detections), crops_per_row) + padding
    canvas_height = (max_crop_height + label_height_estimate + label_spacing + padding) * rows_needed + padding
    
    canvas = Image.new('RGB', (canvas_width, canvas_height), 'black')
    draw = ImageDraw.Draw(canvas)
    
    # Try to load fonts for number display
    try:
        from PIL import ImageFont
        font = ImageFont.truetype("arial.ttf", 24)  # Increased for better visibility
        small_font = ImageFont.truetype("arial.ttf", 12)
    except:
        font = None
        small_font = None
    
    coords_data = {
        'canvas_size': {'width': canvas_width, 'height': canvas_height},
        'images': []
    }
    
    for idx, bbox in enumerate(detections):
        # Crop the detection
        cropped = full_img.crop((
            bbox['xmin'], bbox['ymin'], 
            bbox['xmax'], bbox['ymax']
        ))
        
        # Resize for optimal visibility
        crop_width, crop_height = cropped.size
        
        # Calculate scale factor
        # For small icons, zoom in to make them more visible
        if crop_width < min_display_size and crop_height < min_display_size:
            # Scale up small icons to minimum display size
            scale = min(min_display_size/crop_width, min_display_size/crop_height)
        else:
            # For larger targets, fit within max dimensions
            scale = min(max_crop_width/crop_width, max_crop_height/crop_height, 1.0)
        
        # Apply scaling if needed
        if scale != 1.0:
            new_width = int(crop_width * scale)
            new_height = int(crop_height * scale)
            
            # Ensure we don't exceed max dimensions even when scaling up
            if new_width > max_crop_width or new_height > max_crop_height:
                scale = min(max_crop_width/crop_width, max_crop_height/crop_height)
                new_width = int(crop_width * scale)
                new_height = int(crop_height * scale)
            
            cropped = cropped.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Calculate position on canvas
        row = idx // crops_per_row
        col = idx % crops_per_row
        
        # Calculate label size first
        box_number = idx + 1  # 1-indexed for user friendliness
        label_text = f"{box_number}"
        label_bg_padding = 8  # Increased for more space around number
        
        if font:
            label_bbox = draw.textbbox((0, 0), label_text, font=font)
            text_width = label_bbox[2] - label_bbox[0]
            text_height = label_bbox[3] - label_bbox[1]
            # Make the box square and large enough
            label_size = max(text_width, text_height) + 2 * label_bg_padding
            label_width = label_size
            label_height = label_size
        else:
            label_width = 40  # Fixed size for consistency
            label_height = 40
        x = padding + col * (max_crop_width + padding)
        y = padding + row * (max_crop_height + padding) + label_height + label_spacing
        
        # Paste cropped image
        canvas.paste(cropped, (x, y))
        
        # Draw red border around the cropped image
        border_coords = [x-border_width, y-border_width, 
                        x+cropped.width+border_width, y+cropped.height+border_width]
        draw.rectangle(border_coords, outline='red', width=border_width)
        
        # Position label at top-left corner above the image
        label_x = x
        label_y = y - label_height - label_spacing
        
        # Draw label background with black fill and white border
        draw.rectangle([label_x, label_y, label_x + label_width, label_y + label_height], 
                      fill='black', outline='white', width=3)
        
        # Draw label text centered in the box
        if font:
            label_bbox = draw.textbbox((0, 0), label_text, font=font)
            text_width = label_bbox[2] - label_bbox[0]
            text_height = label_bbox[3] - label_bbox[1]
            text_x = label_x + (label_width - text_width) // 2
            text_y = label_y + (label_height - text_height) // 2
        else:
            text_x = label_x + label_width // 2 - len(label_text) * 5
            text_y = label_y + label_height // 2 - 10
        
        draw.text((text_x, text_y), label_text, fill='white', font=font)
        
        # Get confidence for data storage
        confidence = int(bbox.get('confidence', '0'))
        
        # Add to coordinates data with box number
        coords_data['images'].append({
            'box_number': box_number,
            'id': bbox['id'],
            'label': bbox['label'],
            'confidence': confidence,
            'canvas_coords': {
                'xmin': x, 'ymin': y,
                'xmax': x + cropped.width, 'ymax': y + cropped.height
            },
            'canvas_coords_with_border': {
                'xmin': border_coords[0], 'ymin': border_coords[1],
                'xmax': border_coords[2], 'ymax': border_coords[3]
            },
            'original_screen_coords': {
                'xmin': bbox['xmin'], 'ymin': bbox['ymin'],
                'xmax': bbox['xmax'], 'ymax': bbox['ymax']
            }
        })
    
    # Save canvas
    canvas_path = os.path.join(CROPPED_FOLDER, f"cropped_canvas_{timestamp}.png")
    canvas.save(canvas_path)
    
    # Save coordinates mapping
    coords_path = os.path.join(CROPPED_FOLDER, f"cropped_coords_{timestamp}.json")
    with open(coords_path, 'w') as f:
        json.dump(coords_data, f, indent=2)
    
    if verbose:
        print(f"Cropped canvas saved: {canvas_path}")
        print(f"Coordinates saved: {coords_path}")
    
    return canvas_path


def create_numbered_annotated_screenshot(detections: List[Dict],
                                        full_resolution: Tuple[int, int],
                                        timestamp: str,
                                        verbose: bool = True) -> Optional[str]:
    """Create an annotated screenshot with numbered bounding boxes."""
    if not detections:
        return None
    
    full_img_path = os.path.join(SCREENSHOTS_FOLDER, "screenshot_full.png")
    annotated_img = Image.open(full_img_path).copy()
    draw = ImageDraw.Draw(annotated_img)
    
    # Try to load a font
    try:
        from PIL import ImageFont
        font = ImageFont.truetype("arial.ttf", 24)  # Increased for better visibility
        small_font = ImageFont.truetype("arial.ttf", 14)
    except:
        font = None
        small_font = None
    
    # Draw each detection with its number
    for idx, bbox in enumerate(detections):
        box_number = idx + 1
        x1, y1 = bbox['xmin'], bbox['ymin']
        x2, y2 = bbox['xmax'], bbox['ymax']
        
        # Color based on confidence
        confidence = int(bbox.get('confidence', '0'))
        if confidence >= 90:
            color = '#4CAF50'  # Green
        elif confidence >= 80:
            color = '#FFC107'  # Yellow
        else:
            color = '#F44336'  # Red
        
        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Draw number label with background
        label_text = f"{box_number}"
        label_bg_padding = 8  # Increased for more space
        
        if font:
            label_bbox = draw.textbbox((0, 0), label_text, font=font)
            text_width = label_bbox[2] - label_bbox[0]
            text_height = label_bbox[3] - label_bbox[1]
            # Make the box square and large enough
            label_size = max(text_width, text_height) + 2 * label_bg_padding
            label_width = label_size
            label_height = label_size
        else:
            label_width = 40  # Fixed size
            label_height = 40
        
        # Position label at top-left corner of bbox
        label_x = x1
        label_y = y1 - label_height - 5
        
        # Make sure label is within image bounds
        if label_y < 0:
            label_y = y1 + 5
        
        # Draw label background with black fill
        draw.rectangle([label_x, label_y, label_x + label_width, label_y + label_height], 
                      fill='black', outline='white', width=3)
        
        # Draw label text centered
        if font:
            label_bbox = draw.textbbox((0, 0), label_text, font=font)
            text_width = label_bbox[2] - label_bbox[0]
            text_height = label_bbox[3] - label_bbox[1]
            text_x = label_x + (label_width - text_width) // 2
            text_y = label_y + (label_height - text_height) // 2
        else:
            text_x = label_x + label_width // 2 - len(label_text) * 5
            text_y = label_y + label_height // 2 - 10
            
        draw.text((text_x, text_y), label_text, fill='white', font=font)
        
        # Draw confidence text if space allows
        if bbox.get('confidence') and (x2 - x1) > 100:
            conf_text = f"{confidence}%"
            draw.text((x1 + 5, y1 + 5), conf_text, fill=color, font=small_font)
    
    # Save annotated screenshot
    annotated_path = os.path.join(ANNOTATED_FOLDER, f"annotated_numbered_{timestamp}.png")
    annotated_img.save(annotated_path)
    
    if verbose:
        print(f"Numbered annotated screenshot saved: {annotated_path}")
    
    return annotated_path