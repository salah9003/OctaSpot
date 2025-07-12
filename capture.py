#!/usr/bin/env python3
"""
Octant capture with overlapping regions to prevent boundary splitting.
This module provides functions for capturing and dividing screenshots into overlapping octants.
"""

import mss
import mss.tools
import os
from PIL import Image
from typing import Optional, Tuple, List, Dict


def capture_overlapping_octants(monitor_id: int = 2,
                               overlap_percentage: float = 0.15,
                               target_resolution: Optional[Tuple[int, int]] = None,
                               upscale: bool = True,
                               screenshots_folder: str = "screenshots",
                               verbose: bool = True) -> Optional[Dict]:
    """
    Capture screenshot and save as overlapping octant images to prevent boundary splitting.
    
    The screen is divided into a 4x2 grid with overlapping regions:
    - Each octant overlaps with its neighbors by a specified percentage
    - This ensures objects at boundaries are fully captured in at least one octant
    
    Args:
        monitor_id: Monitor to capture (0=all, 1=primary, 2=second, etc.)
        overlap_percentage: How much octants should overlap (0.0-0.5, default 0.15 = 15%)
        target_resolution: Target resolution for each octant image (width, height)
        upscale: Whether to upscale octants to target resolution
        screenshots_folder: Directory to save screenshots
        verbose: Whether to print status messages
        
    Returns:
        Dictionary containing:
        - 'octant_files': List of octant filenames
        - 'octant_regions': List of dictionaries with octant region info
        - 'full_resolution': Tuple of full screen resolution
    """
    os.makedirs(screenshots_folder, exist_ok=True)
    
    with mss.mss() as sct:
        monitors = sct.monitors
        
        if monitor_id >= len(monitors):
            if verbose:
                print(f"Error: Monitor ID {monitor_id} not found.")
            return None
        
        target_monitor = monitors[monitor_id]
        screenshot = sct.grab(target_monitor)
        
        img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
        
        full_filename = os.path.join(screenshots_folder, "screenshot_full.png")
        img.save(full_filename)
        if verbose:
            print(f"Full screenshot saved: {full_filename}")
        
        width, height = img.size
        
        # Calculate base octant dimensions (without overlap)
        base_octant_width = width // 4
        base_octant_height = height // 2
        
        # Calculate overlap in pixels
        overlap_x = int(base_octant_width * overlap_percentage)
        overlap_y = int(base_octant_height * overlap_percentage)
        
        # Calculate actual octant dimensions (with overlap)
        octant_width = base_octant_width + (2 * overlap_x)
        octant_height = base_octant_height + (2 * overlap_y)
        
        if target_resolution is None and upscale:
            scale_factor = 2
            target_resolution = (octant_width * scale_factor, octant_height * scale_factor)
        elif not upscale:
            target_resolution = (octant_width, octant_height)
        
        octant_files = []
        octant_regions = []
        
        for row in range(2):
            for col in range(4):
                octant_id = row * 4 + col
                
                # Calculate position with overlap
                # First/last columns/rows don't extend beyond screen boundaries
                x_start = max(0, col * base_octant_width - overlap_x)
                y_start = max(0, row * base_octant_height - overlap_y)
                x_end = min(width, (col + 1) * base_octant_width + overlap_x)
                y_end = min(height, (row + 1) * base_octant_height + overlap_y)
                
                # Crop the octant with overlap
                octant_img = img.crop((x_start, y_start, x_end, y_end))
                
                # Resize if target resolution is specified
                if target_resolution:
                    octant_img = octant_img.resize(target_resolution, Image.Resampling.LANCZOS)
                
                # Save the octant
                octant_filename = os.path.join(screenshots_folder, f"octant_{octant_id:02d}.png")
                octant_img.save(octant_filename)
                octant_files.append(octant_filename)
                
                # Store region info for coordinate mapping
                octant_regions.append({
                    'octant_id': octant_id,
                    'screen_region': {
                        'x_start': x_start,
                        'y_start': y_start,
                        'x_end': x_end,
                        'y_end': y_end,
                        'width': x_end - x_start,
                        'height': y_end - y_start
                    },
                    'base_position': {
                        'row': row,
                        'col': col,
                        'base_x': col * base_octant_width,
                        'base_y': row * base_octant_height
                    },
                    'overlap': {
                        'x': overlap_x,
                        'y': overlap_y,
                        'percentage': overlap_percentage
                    },
                    'output_resolution': octant_img.size
                })
                
                if verbose:
                    print(f"Octant {octant_id}: {octant_filename} "
                          f"(region: {x_start},{y_start} to {x_end},{y_end})")
        
        # Save region mapping for coordinate transformation
        import json
        mapping_file = os.path.join(screenshots_folder, "octant_mapping.json")
        with open(mapping_file, 'w') as f:
            json.dump({
                'full_resolution': {'width': width, 'height': height},
                'overlap_percentage': overlap_percentage,
                'octant_regions': octant_regions
            }, f, indent=2)
        
        if verbose:
            print(f"\nOctant capture complete:")
            print(f"  Full resolution: {width}x{height}")
            print(f"  Base octant size: {base_octant_width}x{base_octant_height}")
            print(f"  Overlap: {overlap_percentage*100}% ({overlap_x}x{overlap_y} pixels)")
            print(f"  Actual octant size: {octant_width}x{octant_height}")
            if target_resolution:
                print(f"  Output resolution: {target_resolution[0]}x{target_resolution[1]}")
            print(f"  Mapping saved: {mapping_file}")
        
        return {
            'octant_files': octant_files,
            'octant_regions': octant_regions,
            'full_resolution': (width, height),
            'mapping_file': mapping_file
        }


def map_overlapping_coordinates(octant_id: int, 
                               x_norm: int, 
                               y_norm: int,
                               mapping_file: str) -> Tuple[int, int]:
    """
    Map normalized coordinates from an overlapping octant back to screen coordinates.
    
    Args:
        octant_id: ID of the octant (0-7)
        x_norm: Normalized X coordinate (0-1000) within the octant
        y_norm: Normalized Y coordinate (0-1000) within the octant
        mapping_file: Path to the octant mapping JSON file
        
    Returns:
        Tuple of (screen_x, screen_y) coordinates
    """
    import json
    
    with open(mapping_file, 'r') as f:
        mapping = json.load(f)
    
    # Find the octant region
    octant_region = None
    for region in mapping['octant_regions']:
        if region['octant_id'] == octant_id:
            octant_region = region
            break
    
    if not octant_region:
        raise ValueError(f"Octant {octant_id} not found in mapping")
    
    # Convert normalized to pixel coordinates within octant
    octant_width = octant_region['output_resolution'][0]
    octant_height = octant_region['output_resolution'][1]
    
    x_in_octant = (x_norm / 1000.0) * octant_width
    y_in_octant = (y_norm / 1000.0) * octant_height
    
    # Map to screen coordinates
    screen_region = octant_region['screen_region']
    
    # Scale from output resolution back to screen region
    x_scale = screen_region['width'] / octant_width
    y_scale = screen_region['height'] / octant_height
    
    screen_x = screen_region['x_start'] + (x_in_octant * x_scale)
    screen_y = screen_region['y_start'] + (y_in_octant * y_scale)
    
    return int(screen_x), int(screen_y)