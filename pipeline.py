#!/usr/bin/env python3
"""
Integrated UI Element Detection Pipeline
Improved accuracy through better prompts, NMS, and validation.
"""

import os
import sys
import json
import shutil
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from google import genai
from google.genai import types
from PIL import Image, ImageDraw

# Import detection modules
from detector import detect_items_bbox

# Import configuration
from config import FLASH_MODEL_CONFIG, DETECT_BEST_BOX_CONFIG, DEFAULT_MONITOR_ID, DEFAULT_TARGET_ITEM

# Configuration
MODEL_ID = FLASH_MODEL_CONFIG["model_id"]
TEMPERATURE = FLASH_MODEL_CONFIG["temperature"]
THINKING_BUDGET = FLASH_MODEL_CONFIG["thinking_budget"]

# Folders
SCREENSHOTS_FOLDER = "screenshots"
DETECTIONS_FOLDER = "detections"
PIPELINE_FOLDER = os.path.join(DETECTIONS_FOLDER, "pipeline")

# Global client variable
client = None


def get_icon_xy(target_item: str, **kwargs) -> Optional[List[int]]:
    """
    Simplified wrapper to detect a UI icon/element and return its XY coordinates.

    Args:
        target_item: Description of the icon/element to detect (e.g., "copy icon").
        **kwargs: Optional parameters to pass to detect_ui_element()
                  (e.g., monitor_id=1, image_path='path/to/image.png', verbose=True).

    Returns:
        List[int] as [x, y] if found, else None.
    """
    # Set defaults for GUI agent use (quiet, point-focused, no viz)
    defaults = {
        'mode': 'point',  # Focus on center point coordinates
        'verbose': False,
        'save_visualization': False,
        'use_overlap': True,
        'overlap_percentage': 0.25,
        'merge_duplicates': True,
        'fallback_fullscreen': True,
    }
    # Override defaults with user-provided kwargs
    params = {**defaults, **kwargs}

    result = detect_ui_element(target_item, **params)

    if result['status'] == 'success' and result.get('center_point'):
        return result['center_point']
    else:
        if params.get('verbose'):
            print(f"Detection failed for '{target_item}': {result.get('error', 'Unknown error')}")
        return None


def setup_api_keys() -> bool:
    """Setup API keys for Gemini."""
    if "GEMINI_API_KEY" not in os.environ:
        print("Error: GEMINI_API_KEY environment variable not set.")
        print("Please set it using: export GEMINI_API_KEY='your-api-key'")
        return False
    
    try:
        global client
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))
        return True
    except Exception as e:
        print(f"Error configuring Gemini API: {e}")
        return False


def detect_best_box_using_both_images(canvas_path: str,
                                    annotated_path: str,
                                    bounding_boxes: List[Dict],
                                    target_item: str,
                                    verbose: bool = True) -> Optional[int]:
    """
    Use both canvas and annotated screenshot to select the best matching box ID.
    
    Args:
        canvas_path: Path to the numbered canvas with cropped regions
        annotated_path: Path to the numbered annotated screenshot
        bounding_boxes: List of detected bounding boxes
        target_item: What to look for
        verbose: Whether to print status messages
        
    Returns:
        Box number (1-indexed) of the best match, or None if no match
    """
    if verbose:
        print("\n=== Context-Aware Box Selection ===")
        print(f"Using canvas: {canvas_path}")
        print(f"Using annotated screenshot: {annotated_path}")
    
    try:
        canvas_img = Image.open(canvas_path)
        annotated_img = Image.open(annotated_path)
    except Exception as e:
        if verbose:
            print(f"Error loading images: {e}")
        return None
    
    # Prompt using both images
    prompt = f"""IMAGE 1 (Canvas): A grid of numbered cropped UI elements (e.g., black-bordered boxes with numbers like 1, 2 in white-on-black labels). Each shows a close-up of a potential match.

IMAGE 2 (Annotated Screenshot): The full screen with matching numbered bounding boxes (same numbers as Image 1) drawn on the original locations.

YOUR TASK: Select the SINGLE BEST box number (1, 2, etc.) that matches "{target_item}". Reason step-by-step, considering both images for context.

STEP-BY-STEP REASONING (think internally):
1. Compare numbers across images: Box 1 in Image 1 corresponds to Box 1 in Image 2.
2. For each numbered element in Image 1, examine its visual details (text, icons, shape).
3. Cross-reference with Image 2: Check the screen position and surrounding context (e.g., is it in a menu bar? Near related elements?).
4. Score matches: Prioritize exact visual+contextual fits for "{target_item}" (e.g., an "X" icon in a window title bar for "close button").
5. If multiples match, choose the most prominent (e.g., largest, highest contrast, or most central).
6. If none match well, return box_number null.

RULES:
- MUST use BOTH images: Visuals from Image 1, positions/context from Image 2.
- Focus on interactive elements; ignore non-matches.
- Handle ambiguities: E.g., for multiple "X" icons, pick the one best fitting "{target_item}" based on context.

EXAMPLES:
- Target "close button": If Box 2 shows an "X" in Image 1 and is in a window corner in Image 2, select 2.
- Output: {{"box_number": 2, "confidence": 95, "reason": "Matches X icon in window title (exact visual and position)"}} 

OUTPUT FORMAT (exact JSON object, no extra text):
{{"box_number": <integer or null>, "confidence": <70-100>, "reason": "<brief explanation>"}}

Return box_number null if no good match."""    
    generation_config = types.GenerateContentConfig(
        temperature=DETECT_BEST_BOX_CONFIG["temperature"],
        response_mime_type="application/json",
        thinking_config=types.ThinkingConfig(thinking_budget=DETECT_BEST_BOX_CONFIG["thinking_budget"])
    )
    
    if verbose:
        print("Analyzing both images with Gemini...")
    
    try:
        # Send images in specific order: canvas first, then annotated screenshot, then prompt
        response = client.models.generate_content(
            model=DETECT_BEST_BOX_CONFIG["model_id"],
            contents=[
                canvas_img,      # Image 1: Canvas with numbered cropped elements
                annotated_img,   # Image 2: Annotated screenshot with numbered boxes
                prompt          # The instruction prompt
            ],
            config=generation_config
        )
        
        result = json.loads(response.text)
        
        # Handle both list and dict formats
        if isinstance(result, list) and len(result) > 0:
            result = result[0]  # Get first item from list
        
        if verbose:
            print(f"\nGemini Response:")
            print(f"  Selected box: {result.get('box_number')}")
            print(f"  Confidence: {result.get('confidence')}%")
            print(f"  Reason: {result.get('reason')}")
        
        box_number = result.get('box_number')
        if box_number and box_number > 0:
            return box_number
        else:
            if verbose:
                print("  No valid box number returned")
            return None
        
    except Exception as e:
        if verbose:
            print(f"Error during detection: {e}")
        return None


def detect_center_points_from_bbox_image(image_path: str, 
                                       bounding_boxes: List[Dict],
                                       target_item: str,
                                       verbose: bool = True) -> Optional[List[Dict]]:
    """
    Detect center points of bounding boxes in the cropped images canvas.
    With better prompts and validation.
    """
    if verbose:
        print("\n=== Point Detection Stage ===")
        print(f"Processing image: {image_path}")
    
    try:
        img = Image.open(image_path)
        if verbose:
            print(f"Image size: {img.size[0]}x{img.size[1]}")
    except Exception as e:
        if verbose:
            print(f"Error loading image: {e}")
        return None
    
    # Prompt with better instructions
    prompt = f"""You are analyzing a canvas grid of cropped UI elements, each in a black-bordered box.

YOUR TASK: Identify the SINGLE BEST cropped element matching "{target_item}", then output its geometric center point (inside the border). Reason step-by-step.

STEP-BY-STEP REASONING (think internally):
1. Scan the grid: Each black-bordered crop shows a UI element.
2. Match visuals to "{target_item}" (e.g., text, icons, shape).
3. Select the best one: Prioritize clear, full-visibility matches.
4. Calculate center: Midway point of the content (not border; aim for clickable area).
5. If no match, return empty array.

RULES:
- Only one output: The strongest match.
- Ignore borders: Point to element center (e.g., middle of an icon).
- Context: Consider UI norms (e.g., button centers are clickable).

EXAMPLES:
- For "close button" (X icon): {{"point": [500, 500], "label": "X close icon center"}}
- For "save icon": {{"point": [300, 400], "label": "Floppy disk save icon center"}}

OUTPUT FORMAT (exact JSON array, no extra text):
[{{"point": [y, x], "label": "<short description>"}}]

- point: [y, x] normalized 0-1000.
- Return [] if no match."""
    
    generation_config = types.GenerateContentConfig(
        temperature=TEMPERATURE,
        response_mime_type="application/json",
        thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET)
    )
    
    if verbose:
        print("Detecting center points with Gemini...")
    
    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=[img, prompt],
            config=generation_config
        )
        
        try:
            parsed_points = json.loads(response.text)
        except json.JSONDecodeError:
            json_str = response.text.strip()
            if "```" in json_str:
                import re
                fence_regex = r"```(?:json)?\s*\n?(.*?)\n?\s*```"
                match = re.search(fence_regex, json_str, re.DOTALL)
                if match:
                    json_str = match.group(1).strip()
            parsed_points = json.loads(json_str)
        
        if verbose:
            print(f"Detected {len(parsed_points)} center points")
        
        # Validate points are within image bounds
        validated_points = []
        for point in parsed_points:
            if 'point' in point:
                y, x = point['point']
                if 0 <= y <= 1000 and 0 <= x <= 1000:
                    validated_points.append(point)
                elif verbose:
                    print(f"Warning: Point ({y}, {x}) out of bounds, skipping")
        
        return validated_points
        
    except Exception as e:
        if verbose:
            print(f"Error during point detection: {e}")
        return None


def map_canvas_points_to_screen(points: List[Dict],
                               coords_path: str,
                               full_resolution: Tuple[int, int],
                               verbose: bool = True) -> List[Dict]:
    """
    Map points from cropped images canvas back to full screen coordinates.
    With better validation and error handling.
    """
    try:
        with open(coords_path, 'r') as f:
            coords_data = json.load(f)
    except Exception as e:
        if verbose:
            print(f"Error loading coordinates file: {e}")
        return []
    
    canvas_width = coords_data['canvas_size']['width']
    canvas_height = coords_data['canvas_size']['height']
    
    mapped_points = []
    
    for idx, point in enumerate(points):
        if 'point' not in point:
            continue
            
        y_norm, x_norm = point['point']
        
        # Convert normalized to canvas pixel coordinates
        canvas_x = int((x_norm / 1000.0) * canvas_width)
        canvas_y = int((y_norm / 1000.0) * canvas_height)
        
        # Find which image contains this point
        found_image = None
        for img_info in coords_data['images']:
            canvas_coords = img_info['canvas_coords_with_border']
            if (canvas_coords['xmin'] <= canvas_x <= canvas_coords['xmax'] and
                canvas_coords['ymin'] <= canvas_y <= canvas_coords['ymax']):
                found_image = img_info
                break
        
        if found_image:
            # Map to relative position within the cropped image
            img_canvas_coords = found_image['canvas_coords']
            img_width = img_canvas_coords['xmax'] - img_canvas_coords['xmin']
            img_height = img_canvas_coords['ymax'] - img_canvas_coords['ymin']
            
            rel_x = (canvas_x - img_canvas_coords['xmin']) / img_width
            rel_y = (canvas_y - img_canvas_coords['ymin']) / img_height
            
            # Clamp to valid range
            rel_x = max(0, min(1, rel_x))
            rel_y = max(0, min(1, rel_y))
            
            # Map to screen coordinates
            screen_coords = found_image['original_screen_coords']
            screen_width = screen_coords['xmax'] - screen_coords['xmin']
            screen_height = screen_coords['ymax'] - screen_coords['ymin']
            
            screen_x = screen_coords['xmin'] + (rel_x * screen_width)
            screen_y = screen_coords['ymin'] + (rel_y * screen_height)
            
            # Ensure within screen bounds
            screen_x = max(0, min(screen_x, full_resolution[0] - 1))
            screen_y = max(0, min(screen_y, full_resolution[1] - 1))
            
            mapped_point = {
                'point': [int(screen_y), int(screen_x)],
                'point_normalized': [
                    int((screen_y / full_resolution[1]) * 1000),
                    int((screen_x / full_resolution[0]) * 1000)
                ],
                'label': point.get('label', found_image.get('label', f'Point {idx + 1}')),
                'image_id': found_image['id'],
                'original_canvas_point': point['point'],
                'confidence': found_image.get('confidence', 100)
            }
            
            mapped_points.append(mapped_point)
            
            if verbose:
                print(f"Point {idx + 1}: Image {found_image['id']} -> "
                      f"Screen ({int(screen_x)}, {int(screen_y)})")
        else:
            if verbose:
                print(f"Warning: Point {idx + 1} at canvas ({canvas_x}, {canvas_y}) "
                      f"not found in any image bounds")
    
    return mapped_points


def create_simple_visualization(bbox: Optional[List[int]],
                              center_point: Optional[List[int]],
                              output_filename: str = "detection_result.png",
                              verbose: bool = True) -> Optional[str]:
    """
    Create a simple visualization with just the main detection (bbox and/or dot).
    
    Args:
        bbox: Main bounding box [x, y, width, height]
        center_point: Center point [x, y]
        output_filename: Output filename for the visualization
        verbose: Whether to print status messages
        
    Returns:
        Path to saved visualization or None if error
    """
    full_screenshot_path = os.path.join(SCREENSHOTS_FOLDER, "screenshot_full.png")
    if not os.path.exists(full_screenshot_path):
        if verbose:
            print("Error: Full screenshot not found")
        return None
    
    try:
        img = Image.open(full_screenshot_path)
        draw = ImageDraw.Draw(img)
        
        # Draw bounding box if provided
        if bbox:
            x, y, w, h = bbox
            # Draw main bbox in green
            draw.rectangle([x, y, x+w, y+h], outline='green', width=3)
            
            # Draw confidence label if we have it
            # (Note: confidence would need to be passed separately if desired)
        
        # Draw center point if provided
        if center_point:
            cx, cy = center_point
            
            # Draw crosshair
            crosshair_size = 20
            draw.line([cx - crosshair_size, cy, cx + crosshair_size, cy], fill='red', width=3)
            draw.line([cx, cy - crosshair_size, cx, cy + crosshair_size], fill='red', width=3)
            
            # Draw dot with white center
            radius = 8
            draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], 
                        fill='red', outline='white', width=2)
            
            # Add inner white dot for visibility
            inner_radius = 3
            draw.ellipse([cx - inner_radius, cy - inner_radius, cx + inner_radius, cy + inner_radius], 
                        fill='white')
        
        # Save result
        output_path = os.path.join(DETECTIONS_FOLDER, output_filename)
        os.makedirs(DETECTIONS_FOLDER, exist_ok=True)
        img.save(output_path)
        
        if verbose:
            print(f"\nDetection visualization saved: {output_path}")
        
        return output_path
        
    except Exception as e:
        if verbose:
            print(f"Error creating simple visualization: {e}")
        return None


def visualize_combined_results(bounding_boxes: List[Dict],
                             center_points: List[Dict],
                             full_resolution: Tuple[int, int],
                             output_filename: str = "combined_detection_result.png",
                             verbose: bool = True) -> Optional[str]:
    """
    Create visualization showing both bounding boxes and center points.
    """
    full_screenshot_path = os.path.join(SCREENSHOTS_FOLDER, "screenshot_full.png")
    if not os.path.exists(full_screenshot_path):
        if verbose:
            print("Error: Full screenshot not found")
        return None
    
    try:
        img = Image.open(full_screenshot_path)
        draw = ImageDraw.Draw(img)
        
        # Draw bounding boxes with confidence colors
        for idx, bbox in enumerate(bounding_boxes):
            x1, y1 = bbox['xmin'], bbox['ymin']
            x2, y2 = bbox['xmax'], bbox['ymax']
            
            # Color based on confidence
            confidence = int(bbox.get('confidence', '0'))
            if confidence >= 90:
                color = 'green'
            elif confidence >= 80:
                color = 'yellow'
            else:
                color = 'red'
            
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # Draw ID label
            label = f"{idx + 1} ({confidence}%)"
            draw.rectangle([x1, y1, x1 + 50, y1 + 20], fill='white', outline=color)
            draw.text((x1 + 5, y1 + 2), label, fill=color)
            
            # Add merged count info
            if bbox.get('merged_count', 0) > 1:
                info = f"M:{bbox['merged_count']}"
                draw.text((x1 + 55, y1 + 2), info, fill=color)
        
        # Draw center points with visualization
        for idx, point in enumerate(center_points):
            y, x = point['point']
            
            # Draw crosshair
            crosshair_size = 20
            draw.line([x - crosshair_size, y, x + crosshair_size, y], fill='blue', width=3)
            draw.line([x, y - crosshair_size, x, y + crosshair_size], fill='blue', width=3)
            
            # Draw circle with gradient effect
            for r in range(12, 6, -1):
                alpha = int(255 * (12 - r) / 6)
                color = (59, 104, 255, alpha)
                draw.ellipse([x - r, y - r, x + r, y + r], 
                            fill=None, outline='blue', width=2)
            
            # Draw center dot
            draw.ellipse([x - 4, y - 4, x + 4, y + 4], 
                        fill='white', outline='blue', width=2)
        
        # Save result
        output_path = os.path.join(PIPELINE_FOLDER, output_filename)
        os.makedirs(PIPELINE_FOLDER, exist_ok=True)
        img.save(output_path)
        
        if verbose:
            print(f"\nCombined visualization saved: {output_path}")
        
        return output_path
        
    except Exception as e:
        if verbose:
            print(f"Error creating visualization: {e}")
        return None


def select_best_detection(bounding_boxes: List[Dict], 
                         center_points: List[Dict],
                         target_item: str) -> Dict:
    """
    Select the best detection from multiple candidates.
    Uses confidence scores and center point alignment.
    """
    if not bounding_boxes:
        return {'bbox': None, 'point': None}
    
    if not center_points:
        # Return highest confidence bbox
        best_bbox = max(bounding_boxes, 
                       key=lambda x: int(x.get('confidence', '0')))
        return {'bbox': best_bbox, 'point': None}
    
    # If we have center points, find the bbox that contains it
    best_point = center_points[0]  # Already filtered to best match
    y, x = best_point['point']
    
    # Find bbox that contains this point
    containing_bbox = None
    for bbox in bounding_boxes:
        if (bbox['xmin'] <= x <= bbox['xmax'] and 
            bbox['ymin'] <= y <= bbox['ymax']):
            containing_bbox = bbox
            break
    
    if containing_bbox:
        return {'bbox': containing_bbox, 'point': best_point}
    
    # If no bbox contains the point, return closest bbox
    min_dist = float('inf')
    closest_bbox = bounding_boxes[0]
    
    for bbox in bounding_boxes:
        # Calculate distance from point to bbox center
        bbox_cx = (bbox['xmin'] + bbox['xmax']) / 2
        bbox_cy = (bbox['ymin'] + bbox['ymax']) / 2
        dist = ((x - bbox_cx) ** 2 + (y - bbox_cy) ** 2) ** 0.5
        
        if dist < min_dist:
            min_dist = dist
            closest_bbox = bbox
    
    return {'bbox': closest_bbox, 'point': best_point}


def detect_ui_element(target_item: str,
                     monitor_id: int = DEFAULT_MONITOR_ID,
                     mode: str = "bbox",
                     verbose: bool = True,
                     use_overlap: bool = True,
                     overlap_percentage: float = 0.25,
                     merge_duplicates: bool = True,
                     fallback_fullscreen: bool = True,
                     image_path: Optional[str] = None,
                     save_visualization: bool = True) -> Dict:
    """
    UI element detection with improved accuracy.
    
    Args:
        target_item: Description of what to detect
        monitor_id: Monitor to capture (ignored if image_path is provided)
        mode: Detection mode - "bbox", "point", or "bbox_and_point"
        verbose: Whether to print status messages
        use_overlap: Whether to use overlapping octants
        overlap_percentage: How much octants should overlap (default: 0.25)
        merge_duplicates: Whether to merge duplicate detections
        fallback_fullscreen: Whether to try full screen if octants fail
        image_path: Optional path to image file to analyze instead of capturing
        save_visualization: Whether to save visualization with detection dot
        
    Returns:
        Dictionary containing detection results with format suitable for ScreenSpot
        
    Note:
        When a bounding box is detected, the center_point is always calculated as
        the geometric center of the bounding box, ensuring the dot appears in the
        middle of the detected element.
    """
    result = {
        'status': 'error',
        'target_item': target_item,
        'bbox': None,
        'center_point': None,
        'all_bounding_boxes': [],
        'all_center_points': [],
        'confidence': 0,
        'method': None,
        'error': None
    }
    
    if not setup_api_keys():
        result['error'] = "API key setup failed"
        raise RuntimeError(result['error'])  # Raise for calling scripts to handle
    
    try:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Detection Pipeline")
            print(f"Target: {target_item}")
            print(f"Mode: {mode}")
            if image_path:
                print(f"Image: {image_path}")
            else:
                print(f"Monitor: {monitor_id}")
            print(f"Overlap: {overlap_percentage*100}%")
            print(f"{'='*60}")
        
        # If image_path is provided, copy it to the expected location
        if image_path:
            if not os.path.exists(image_path):
                result['error'] = f"Image file not found: {image_path}"
                return result
            
            # Create screenshots folder and copy image
            os.makedirs(SCREENSHOTS_FOLDER, exist_ok=True)
            screenshot_path = os.path.join(SCREENSHOTS_FOLDER, "screenshot_full.png")
            
            if verbose:
                print(f"\nUsing provided image: {image_path}")
            
            # Copy the image to the expected location
            shutil.copy2(image_path, screenshot_path)
        
        # Stage 1: Bounding box detection
        bbox_result = detect_items_bbox(
            items=target_item,
            monitor_id=monitor_id,
            verbose=verbose,
            use_overlap=use_overlap,
            overlap_percentage=overlap_percentage,
            merge_duplicates=merge_duplicates,
            fallback_fullscreen=fallback_fullscreen,
            skip_capture=bool(image_path)
        )
        
        if not bbox_result['success'] or not bbox_result['bounding_boxes']:
            result['error'] = "No UI element detected"
            result['method'] = 'octant_overlap' if use_overlap else 'octant'
            return result
        
        result['all_bounding_boxes'] = bbox_result['bounding_boxes']
        result['method'] = 'octant_overlap' if use_overlap else 'octant'
        result['bbox_result'] = bbox_result  # Store full bbox result for access to images
        
        # Get screen resolution
        full_screenshot_path = os.path.join(SCREENSHOTS_FOLDER, "screenshot_full.png")
        with Image.open(full_screenshot_path) as img:
            screen_resolution = img.size
        
        # Stage 2: Context-aware selection using both canvas and annotated screenshot
        canvas_path = bbox_result.get('cropped_images_canvas')
        numbered_annotated_path = bbox_result.get('numbered_annotated_image')
        
        best_box_number = None
        if canvas_path and numbered_annotated_path and os.path.exists(canvas_path) and os.path.exists(numbered_annotated_path):
            # Use both images to select the best box
            best_box_number = detect_best_box_using_both_images(
                canvas_path,
                numbered_annotated_path,
                result['all_bounding_boxes'],
                target_item,
                verbose
            )
            
            if best_box_number and verbose:
                print(f"\nSelected box number: {best_box_number}")
        
        # Get the selected bbox based on box number
        best_bbox = None
        if best_box_number:
            # Box numbers are 1-indexed
            if 0 < best_box_number <= len(result['all_bounding_boxes']):
                best_bbox = result['all_bounding_boxes'][best_box_number - 1]
                result['selected_box_number'] = best_box_number
        
        # Initialize best dict
        best = None
        
        # If context-aware selection didn't work, fall back to original method
        if not best_bbox:
            if mode in ["point", "bbox_and_point"] and canvas_path and os.path.exists(canvas_path):
                # Find the corresponding coordinates file
                canvas_filename = os.path.basename(canvas_path)
                timestamp = canvas_filename.replace('cropped_canvas_', '').replace('.png', '')
                coords_path = os.path.join(DETECTIONS_FOLDER, "cropped", f"cropped_coords_{timestamp}.json")
                
                if os.path.exists(coords_path):
                    detected_points = detect_center_points_from_bbox_image(
                        canvas_path, 
                        result['all_bounding_boxes'],
                        target_item,
                        verbose
                    )
                    
                    if detected_points:
                        mapped_points = map_canvas_points_to_screen(
                            detected_points,
                            coords_path,
                            screen_resolution,
                            verbose
                        )
                        
                        result['all_center_points'] = mapped_points
            
            # Select best detection using original method (for any mode)
            best = select_best_detection(
                result['all_bounding_boxes'],
                result['all_center_points'],
                target_item
            )
            best_bbox = best.get('bbox')
        else:
            # Create a best dict for compatibility
            best = {'bbox': best_bbox, 'point': None}
        
        # Format result for ScreenSpot benchmark
        if best_bbox:
            result['bbox'] = [
                best_bbox['xmin'],
                best_bbox['ymin'],
                best_bbox['xmax'] - best_bbox['xmin'],  # width
                best_bbox['ymax'] - best_bbox['ymin']   # height
            ]
            result['confidence'] = int(best_bbox.get('confidence', '0')) / 100.0
            result['status'] = 'success'
            
            # Always calculate center point from the bounding box
            # This ensures the dot is in the center of the detected box
            bbox_center_x = best_bbox['xmin'] + (best_bbox['xmax'] - best_bbox['xmin']) // 2
            bbox_center_y = best_bbox['ymin'] + (best_bbox['ymax'] - best_bbox['ymin']) // 2
            result['center_point'] = [bbox_center_x, bbox_center_y]
        elif best and best.get('point'):
            # Only use detected point if no bbox was found
            y, x = best['point']['point']
            result['center_point'] = [x, y]
        
        # Create visualization if requested
        if verbose and mode == "bbox_and_point":
            combined_viz = visualize_combined_results(
                result['all_bounding_boxes'],
                result['all_center_points'],
                screen_resolution,
                verbose=verbose
            )
        
        # Create simple visualization with final detection
        if save_visualization and result['status'] == 'success':
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            viz_filename = f"detection_{timestamp}.png"
            
            # Create visualization with the detected target
            viz_path = create_simple_visualization(
                bbox=result['bbox'],
                center_point=result['center_point'],
                output_filename=viz_filename,
                verbose=verbose
            )
            
            if viz_path:
                result['visualization_path'] = viz_path
                
                # Also save with a predictable name for easy access
                latest_viz_path = os.path.join(DETECTIONS_FOLDER, "latest_detection.png")
                try:
                    shutil.copy2(viz_path, latest_viz_path)
                    if verbose:
                        print(f"Latest detection also saved as: {latest_viz_path}")
                except:
                    pass
        
        # Save pipeline results
        save_pipeline_results(result, verbose)
        
    except Exception as e:
        result['error'] = str(e)
        if verbose:
            print(f"\nPipeline error: {e}")
            import traceback
            traceback.print_exc()
    
    return result


def save_pipeline_results(result: Dict, verbose: bool = True):
    """Save comprehensive pipeline results to JSON file."""
    os.makedirs(PIPELINE_FOLDER, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"pipeline_results_{timestamp}.json"
    filepath = os.path.join(PIPELINE_FOLDER, filename)
    
    latest_filepath = os.path.join(PIPELINE_FOLDER, "latest_results.json")
    
    try:
        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2)
        
        shutil.copy2(filepath, latest_filepath)
        
        if verbose:
            print(f"\nResults saved:")
            print(f"  - {filepath}")
            print(f"  - {latest_filepath}")
            
    except Exception as e:
        if verbose:
            print(f"Error saving results: {e}")


def print_results_summary(result: Dict):
    """Print a formatted summary of detection results."""
    print(f"\n{'='*50}")
    print("Detection Results Summary")
    print(f"{'='*50}")
    
    print(f"Target: {result['target_item']}")
    print(f"Status: {result['status']}")
    print(f"Method: {result.get('method', 'unknown')}")
    
    if result['error']:
            print(f"Error: {result['error']}")
            return
    
    if result['bbox']:
        x, y, w, h = result['bbox']
        print(f"\nBest Bounding Box:")
        print(f"  Position: ({x}, {y})")
        print(f"  Size: {w} x {h}")
        print(f"  Confidence: {result['confidence']*100:.1f}%")
    
    if result['center_point']:
        x, y = result['center_point']
        print(f"\nCenter Point: ({x}, {y})")
    
    print(f"\nTotal Detections:")
    print(f"  Bounding boxes: {len(result.get('all_bounding_boxes', []))}")
    print(f"  Center points: {len(result.get('all_center_points', []))}")


def main():
    """Command-line interface for the detection pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="UI Element Detection Pipeline"
    )
    parser.add_argument(
        "target",
        nargs="?",
        default=DEFAULT_TARGET_ITEM,
        help="Description of UI element to detect"
    )
    parser.add_argument(
        "--monitor",
        type=int,
        default=DEFAULT_MONITOR_ID,
        help=f"Monitor ID to capture (default: {DEFAULT_MONITOR_ID})"
    )
    parser.add_argument(
        "--mode",
        choices=["bbox", "point", "bbox_and_point"],
        default="bbox",
        help="Detection mode (default: bbox)"
    )
    parser.add_argument(
        "--no-overlap",
        action="store_true",
        help="Disable overlapping octants"
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.25,
        help="Overlap percentage (0.0-0.5, default: 0.25)"
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Disable duplicate merging"
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Disable fullscreen fallback"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to image file to analyze"
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Disable saving visualization with detection"
    )
    
    args = parser.parse_args()
    
    # Run detection
    result = detect_ui_element(
        target_item=args.target,
        monitor_id=args.monitor,
        mode=args.mode,
        verbose=not args.quiet,
        use_overlap=not args.no_overlap,
        overlap_percentage=args.overlap,
        merge_duplicates=not args.no_merge,
        fallback_fullscreen=not args.no_fallback,
        image_path=args.image,
        save_visualization=not args.no_viz
    )
    
    # Print summary
    print_results_summary(result)
    
    sys.exit(0 if result['status'] == 'success' else 1)


if __name__ == "__main__":
    main()