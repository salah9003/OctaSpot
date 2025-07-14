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
from config import DETECT_BEST_BOX_CONFIG, PRECISION_POINT_CONFIG, DEFAULT_MONITOR_ID, DEFAULT_TARGET_ITEM

# Folders
SCREENSHOTS_FOLDER = "screenshots"
DETECTIONS_FOLDER = "detections"
PIPELINE_FOLDER = os.path.join(DETECTIONS_FOLDER, "pipeline")
LATEST_RESULTS_FOLDER = "latest_results"

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
    Returns "target" if the target item is not found in any box.
    
    Args:
        canvas_path: Path to the numbered canvas with cropped regions
        annotated_path: Path to the numbered annotated screenshot
        bounding_boxes: List of detected bounding boxes
        target_item: What to look for
        verbose: Whether to print status messages
        
    Returns:
        Box number (1-indexed) of the best match, "target" if not found in boxes, or None if error
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

YOUR TASK: Select the SINGLE BEST box number (1, 2, etc.) that matches "{target_item}". This selected box will be used later to precisely locate the exact point of the target within it, so prioritize boxes that provide sufficient context and size for accurate sub-element detection. Reason step-by-step, considering both images for context.

STEP-BY-STEP REASONING (think internally):
1. Compare numbers across images: Box 1 in Image 1 corresponds to Box 1 in Image 2.
2. For each numbered element in Image 1, examine its visual details (text, icons, shape).
3. Cross-reference with Image 2: Check the screen position and surrounding context (e.g., is it in a menu bar? Near related elements?).
4. Score matches: Prioritize exact visual+contextual fits for "{target_item}" (e.g., an "X" icon in a window title bar for "close button"). For sub-elements (e.g., "ear" in a portrait), prefer larger boxes that include the full containing object (e.g., the whole head or body) for better precision later, rather than tiny crops of just the sub-element.
5. If multiples match, choose the most prominent and suitable for precise pointing (e.g., largest, highest contrast, most central, or with best surrounding context).
6. If none match well in the boxes but the target exists elsewhere in the screenshot (Image 2), return point_at with what to search for.
7. The Box MUST contain the target item, and NOT the box that is 'Nearest to the item'.


RULES:
- MUST use BOTH images: Visuals from Image 1, positions/context from Image 2.
- Focus on interactive elements; ignore non-matches.
- Handle ambiguities: E.g., for multiple "X" icons, pick the one best fitting "{target_item}" based on context. For targets like "Putin's ear", select the largest box containing Putin's full body or head that includes the ear, to enable precise ear location within it later.
- If the target exists in the full screenshot (Image 2) but not in any of the numbered boxes, return point_at instead of box_number.
- CAVAS BOXES POSITIONS ARE NOT TO BE USED AS A REFERENCE FOR THE TARGET LOCATION.

EXAMPLES:
- Output: {{"box_number": 2, "confidence": 95, "reason": "Matches X icon in window title (exact visual and position)"}} 
- Output: {{"box_number": 2, "confidence": 93, "reason": "Box 3 contains a small picture of an eye, however Box 2 contains the entire head, which includes the eye and provides better context for precise pointing"}} 
- Output: {{"box_number": 4, "confidence": 98, "reason": "Box 4 shows Putin's full body including the ear, larger and clearer than Box 5 which only crops the head, enabling better precise ear detection later"}}
- Output: {{"point_at": "The 2nd Icon located in the taskbar, with a red Lion face", "confidence": 85, "reason": "The brave icon is visible in the taskbar, but not show in the canvas"}}
- Output: {{"box_number": null, "confidence": 0, "reason": "Target not found anywhere in the screenshot"}}

OUTPUT FORMAT (exact JSON object, no extra text):
{{"box_number": <integer or null>, "confidence": <70-100>, "reason": "<detailed reasoning explanation>"}} OR
{{"point_at": "<detailed description of what to search for>", "confidence": <70-100>, "reason": "<brief explanation>"}}

Return box_number null if target doesn't exist anywhere."""
    
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
        elif result.get('point_at'):
            # Store the point_at value to be used by fullscreen detection
            return {"type": "target", "point_at": result.get('point_at')}
        else:
            if verbose:
                print("  No valid box number returned and no target found")
            return None
        
    except Exception as e:
        if verbose:
            print(f"Error during detection: {e}")
        return None


def detect_precise_point_in_crop(bbox: Dict,
                                target_item: str,
                                full_screenshot_path: str,
                                verbose: bool = True) -> Optional[Dict]:
    """
    Extract crop from bbox and detect precise point for sub-element targets.
    
    Args:
        bbox: Selected bounding box with xmin, ymin, xmax, ymax
        target_item: What to find precisely (e.g., "Putin's ear")
        full_screenshot_path: Path to original screenshot
        verbose: Whether to print status messages
        
    Returns:
        Dict with normalized point coordinates or None if failed
    """
    if verbose:
        print("\n=== Precision Point Detection ===")
        print(f"Extracting crop for precise detection of: {target_item}")
    
    try:
        # Load the full screenshot
        full_img = Image.open(full_screenshot_path)
        
        # Extract the crop using bbox coordinates
        crop = full_img.crop((
            bbox['xmin'], 
            bbox['ymin'], 
            bbox['xmax'], 
            bbox['ymax']
        ))
        
        if verbose:
            print(f"Cropped region: {crop.size[0]}x{crop.size[1]} pixels")
        
        # Save the cropped image for debugging/visualization
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            crop_filename = f"precision_crop_{timestamp}.png"
            crop_path = os.path.join(LATEST_RESULTS_FOLDER, crop_filename)
            os.makedirs(LATEST_RESULTS_FOLDER, exist_ok=True)
            crop.save(crop_path)
            if verbose:
                print(f"Saved cropped region: {crop_path}")
        except Exception as e:
            if verbose:
                print(f"Warning: Could not save crop: {e}")
        
        # Create precision detection prompt
        prompt = f"""point at the target item: '{target_item}' in the image

OUTPUT FORMAT (exact JSON, no extra text):
{{"point": [x, y], "confidence": <70-100>, "description": "<what you found>"}}

Where:
- point: [x, y] normalized 0-1000 
- confidence: How certain you are this is the correct {target_item}
- description: Brief description of what you're pointing at

Return null if {target_item} is not clearly visible in the image."""

        generation_config = types.GenerateContentConfig(
            temperature=PRECISION_POINT_CONFIG["temperature"],
            response_mime_type="application/json",
            thinking_config=types.ThinkingConfig(thinking_budget=PRECISION_POINT_CONFIG["thinking_budget"])
        )
        
        if verbose:
            print("Analyzing crop for precise point detection...")
            print(f"Using model: {PRECISION_POINT_CONFIG['model_id']}")
        
        response = client.models.generate_content(
            model=PRECISION_POINT_CONFIG["model_id"],
            contents=[crop, prompt],
            config=generation_config
        )
        
        result = json.loads(response.text)
        
        if result and result.get('point'):
            confidence = result.get('confidence', 0)
            
            # Check if confidence meets minimum threshold
            if confidence < PRECISION_POINT_CONFIG.get('min_confidence', 80):
                if verbose:
                    print(f"Precision confidence {confidence}% below threshold {PRECISION_POINT_CONFIG['min_confidence']}%")
                return None
            
            if verbose:
                print(f"Precise point found: {result['description']}")
                print(f"Confidence: {confidence}%")
                print(f"Normalized coordinates: ({result['point'][0]}, {result['point'][1]})")
            return result
        else:
            if verbose:
                print("No precise point detected in crop")
            return None
            
    except Exception as e:
        if verbose:
            print(f"Error during precision detection: {e}")
        return None


def map_crop_point_to_screen(crop_point: List[int],
                            bbox: Dict,
                            normalized: bool = True,
                            verbose: bool = False) -> List[int]:
    """
    Map a point from crop coordinates to screen coordinates.
    
    Args:
        crop_point: [x, y] point in crop coordinates
        bbox: Bounding box with screen coordinates
        normalized: Whether crop_point is normalized (0-1000) or pixel coordinates
        
    Returns:
        [x, y] in screen coordinates
    """
    y_crop, x_crop = crop_point
    
    # Convert from normalized to pixel coordinates if needed
    if normalized:
        crop_width = bbox['xmax'] - bbox['xmin']
        crop_height = bbox['ymax'] - bbox['ymin']
        x_pixel = (x_crop / 1000.0) * crop_width
        y_pixel = (y_crop / 1000.0) * crop_height
    else:
        x_pixel = x_crop
        y_pixel = y_crop
    
    # Add bbox offset to get screen coordinates
    screen_x = int(bbox['xmin'] + x_pixel)
    screen_y = int(bbox['ymin'] + y_pixel)
    
    # Debug logging
    if verbose:
        print(f"\nDEBUG: Mapping crop point to screen:")
        print(f"  Crop point (normalized): {crop_point}")
        print(f"  Crop point (pixels): ({x_pixel:.1f}, {y_pixel:.1f})")
        print(f"  BBox: xmin={bbox['xmin']}, ymin={bbox['ymin']}, xmax={bbox['xmax']}, ymax={bbox['ymax']}")
        print(f"  BBox size: {crop_width}x{crop_height}")
        print(f"  Screen point: ({screen_x}, {screen_y})")
    
    return [screen_x, screen_y]


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
    Uses confidence scores to select the best bounding box.
    
    Note: center_points parameter is kept for compatibility but is no longer used.
    """
    if not bounding_boxes:
        return {'bbox': None, 'point': None}
    
    # Return highest confidence bbox
    best_bbox = max(bounding_boxes, 
                   key=lambda x: int(x.get('confidence', '0')))
    return {'bbox': best_bbox, 'point': None}


def detect_ui_element(target_item: str,
                     monitor_id: int = DEFAULT_MONITOR_ID,
                     mode: str = "point",
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
            target=target_item,
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
        if best_box_number and isinstance(best_box_number, int):
            # Box numbers are 1-indexed
            if 0 < best_box_number <= len(result['all_bounding_boxes']):
                best_bbox = result['all_bounding_boxes'][best_box_number - 1]
                result['selected_box_number'] = best_box_number
        elif best_box_number and isinstance(best_box_number, dict) and best_box_number.get('type') == 'target':
            # Target found in fullscreen but not in detected boxes
            # Use fullscreen detection
            if verbose:
                print("\nTarget exists in fullscreen but not in detected boxes.")
                print("Using direct fullscreen detection...")
            
            fullscreen_result = detect_target_in_fullscreen(
                best_box_number.get('point_at'),
                full_screenshot_path,
                verbose
            )
            
            if fullscreen_result:
                result['center_point'] = fullscreen_result['point']
                result['confidence'] = fullscreen_result['confidence'] / 100.0
                result['status'] = 'success'
                result['method'] = 'fullscreen_direct'
                result['precision_mode'] = True
                result['precision_confidence'] = fullscreen_result['confidence']
                result['precision_description'] = fullscreen_result['description']
                
                # Create simple visualization with final detection
                if save_visualization and result['status'] == 'success':
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    viz_filename = f"detection_{timestamp}.png"
                    
                    viz_path = create_simple_visualization(
                        bbox=None,
                        center_point=result['center_point'],
                        output_filename=viz_filename,
                        verbose=verbose
                    )
                    
                    if viz_path:
                        result['visualization_path'] = viz_path
                
                # Save to latest_results
                if result['status'] == 'success':
                    os.makedirs(LATEST_RESULTS_FOLDER, exist_ok=True)
                    
                    if 'visualization_path' in result and os.path.exists(result['visualization_path']):
                        latest_result_path = os.path.join(LATEST_RESULTS_FOLDER, "latest_result.png")
                        try:
                            shutil.copy2(result['visualization_path'], latest_result_path)
                            if verbose:
                                print(f"\nLatest result image saved as: {latest_result_path}")
                        except Exception as e:
                            if verbose:
                                print(f"Error saving latest result: {e}")
                
                return result
        
        # Initialize best dict
        best = None
        
        # If context-aware selection didn't work, fall back to original method
        if not best_bbox:
            # Select best detection using original method (for any mode)
            best = select_best_detection(
                result['all_bounding_boxes'],
                [],  # No center points after removal of detection function
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
            
            # Check if we need precision point detection
            # This is useful when the target is a sub-element (e.g., "ear" within a face)
            use_precision = mode in ["point", "bbox_and_point"]
            
            if use_precision and best_bbox:
                # Try precision detection on the selected crop
                full_screenshot_path = os.path.join(SCREENSHOTS_FOLDER, "screenshot_full.png")
                precision_result = detect_precise_point_in_crop(
                    best_bbox,
                    target_item,
                    full_screenshot_path,
                    verbose
                )
                
                if precision_result and precision_result.get('point'):
                    # Map the precise point to screen coordinates
                    precise_screen_point = map_crop_point_to_screen(
                        precision_result['point'],
                        best_bbox,
                        normalized=True,
                        verbose=verbose
                    )
                    result['center_point'] = precise_screen_point
                    result['precision_mode'] = True
                    result['precision_confidence'] = precision_result.get('confidence', 0)
                    result['precision_description'] = precision_result.get('description', '')
                    
                    if verbose:
                        print(f"\nPrecision point mapped to screen: ({precise_screen_point[0]}, {precise_screen_point[1]})")
            
            # Fall back to bbox center if precision detection wasn't used or failed
            if not result.get('center_point'):
                # Calculate center point from the bounding box
                bbox_center_x = best_bbox['xmin'] + (best_bbox['xmax'] - best_bbox['xmin']) // 2
                bbox_center_y = best_bbox['ymin'] + (best_bbox['ymax'] - best_bbox['ymin']) // 2
                result['center_point'] = [bbox_center_x, bbox_center_y]
                result['precision_mode'] = False
        
        # Create visualization if requested
        if verbose and mode == "bbox_and_point":
            visualize_combined_results(
                result['all_bounding_boxes'],
                [],  # No center points after removal of detection function
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
        
        # Save all latest images to latest_results folder
        if result['status'] == 'success':
            # Create latest_results folder
            os.makedirs(LATEST_RESULTS_FOLDER, exist_ok=True)
            
            # Save latest result/detection image
            if 'visualization_path' in result and os.path.exists(result['visualization_path']):
                latest_result_path = os.path.join(LATEST_RESULTS_FOLDER, "latest_result.png")
                try:
                    shutil.copy2(result['visualization_path'], latest_result_path)
                    if verbose:
                        print(f"\nLatest result image saved as: {latest_result_path}")
                except Exception as e:
                    if verbose:
                        print(f"Error saving latest result: {e}")
            
            # Save latest canvas and annotated images
            if 'bbox_result' in result:
                bbox_result = result['bbox_result']
                
                # Save latest canvas image
                if bbox_result.get('cropped_images_canvas') and os.path.exists(bbox_result['cropped_images_canvas']):
                    latest_canvas_path = os.path.join(LATEST_RESULTS_FOLDER, "latest_canvas.png")
                    try:
                        shutil.copy2(bbox_result['cropped_images_canvas'], latest_canvas_path)
                        if verbose:
                            print(f"Latest canvas image saved as: {latest_canvas_path}")
                    except Exception as e:
                        if verbose:
                            print(f"Error saving latest canvas: {e}")
                
                # Save latest annotated image
                if bbox_result.get('numbered_annotated_image') and os.path.exists(bbox_result['numbered_annotated_image']):
                    latest_annotated_path = os.path.join(LATEST_RESULTS_FOLDER, "latest_annotated.png")
                    try:
                        shutil.copy2(bbox_result['numbered_annotated_image'], latest_annotated_path)
                        if verbose:
                            print(f"Latest annotated image saved as: {latest_annotated_path}")
                    except Exception as e:
                        if verbose:
                            print(f"Error saving latest annotated: {e}")
        
        # Save pipeline results
        save_pipeline_results(result, verbose)
        
    except Exception as e:
        result['error'] = str(e)
        if verbose:
            print(f"\nPipeline error: {e}")
            import traceback
            traceback.print_exc()
    
    return result


def detect_target_in_fullscreen(target_item: str,
                               screenshot_path: str,
                               verbose: bool = True) -> Optional[Dict]:
    """
    Detect target directly in full screenshot using Gemini.
    
    Args:
        target_item: What to find
        screenshot_path: Path to full screenshot
        verbose: Whether to print status messages
        
    Returns:
        Dict with point coordinates or None if failed
    """
    if verbose:
        print("\n=== Fullscreen Target Detection ===")
        print(f"Target: {target_item}")
    
    try:
        img = Image.open(screenshot_path)
        img_width, img_height = img.size
        
        if verbose:
            print(f"Screenshot size: {img_width}x{img_height}")
    except Exception as e:
        if verbose:
            print(f"Error loading screenshot: {e}")
        return None
    
    prompt = f"""Point to: {target_item}

The answer should follow the json format:
{{"point": [x, y], "confidence": <70-100>, "description": "<what you found>"}}

Where:
- point: [x, y] format normalized to 0-1000.
- confidence: How certain you are this is the correct {target_item}
- description: Brief description of what you found

Return null if {target_item} is not found."""

    generation_config = types.GenerateContentConfig(
        temperature=PRECISION_POINT_CONFIG["temperature"],
        response_mime_type="application/json",
        thinking_config=types.ThinkingConfig(thinking_budget=PRECISION_POINT_CONFIG["thinking_budget"])
    )
    
    if verbose:
        print("Analyzing full screenshot with Gemini...")
    
    try:
        response = client.models.generate_content(
            model=PRECISION_POINT_CONFIG["model_id"],
            contents=[img, prompt],
            config=generation_config
        )
        
        result = json.loads(response.text)
        
        if result and result.get('point'):
            confidence = result.get('confidence', 0)
            
            if confidence < PRECISION_POINT_CONFIG.get('min_confidence', 80):
                if verbose:
                    print(f"Confidence {confidence}% below threshold")
                return None
            
            # Convert normalized coordinates to screen coordinates
            x_norm, y_norm = result['point']
            screen_x = int((x_norm / 1000.0) * img_width)
            screen_y = int((y_norm / 1000.0) * img_height)
            
            if verbose:
                print(f"\nTarget found in fullscreen:")
                print(f"  Description: {result['description']}")
                print(f"  Confidence: {confidence}%")
                print(f"  Screen coordinates: ({screen_x}, {screen_y})")
            
            return {
                'point': [screen_x, screen_y],
                'point_normalized': result['point'],
                'confidence': confidence,
                'description': result['description'],
                'method': 'fullscreen_direct'
            }
        else:
            if verbose:
                print("No target found in fullscreen")
            return None
            
    except Exception as e:
        if verbose:
            print(f"Error during fullscreen detection: {e}")
        return None


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
        
        # Show precision mode info if available
        if result.get('precision_mode'):
            print(f"  Precision Mode: Enabled")
            print(f"  Precision Target: {result.get('precision_description', 'N/A')}")
            print(f"  Precision Confidence: {result.get('precision_confidence', 0)}%")
        else:
            print(f"  Precision Mode: Disabled (using bbox center)")
    
    print(f"\nTotal Detections:")
    print(f"  Bounding boxes: {len(result.get('all_bounding_boxes', []))}")


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
        default="bbox_and_point",
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