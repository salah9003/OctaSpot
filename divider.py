#!/usr/bin/env python3
"""
Helper function to divide an existing image into overlapping octants
"""

import os
import json
from PIL import Image
from typing import Optional, Tuple, List, Dict


def divide_image_into_overlapping_octants(
    full_img: Image.Image,
    overlap_percentage: float = 0.15,
    target_resolution: Optional[Tuple[int, int]] = None,
    upscale: bool = True,
    screenshots_folder: str = "screenshots",
    verbose: bool = True
) -> Optional[Dict]:
    """
    Divide an existing image into overlapping octants.
    
    Args:
        full_img: PIL Image object to divide
        overlap_percentage: How much octants should overlap (0.0-0.5)
        target_resolution: Target resolution for each octant (width, height)
        upscale: Whether to upscale octants to target resolution
        screenshots_folder: Directory to save octant images
        verbose: Whether to print status messages
        
    Returns:
        Dictionary containing octant files and mapping data
    """
    if overlap_percentage < 0 or overlap_percentage > 0.5:
        if verbose:
            print(f"Error: overlap_percentage must be between 0.0 and 0.5")
        return None
    
    width, height = full_img.size
    
    # Calculate octant dimensions with overlap
    octant_width = width // 4
    octant_height = height // 2
    
    # Calculate overlap in pixels
    overlap_x = int(octant_width * overlap_percentage)
    overlap_y = int(octant_height * overlap_percentage)
    
    # Adjust octant size to include overlap
    expanded_width = octant_width + overlap_x
    expanded_height = octant_height + overlap_y
    
    if verbose:
        print(f"Dividing {width}x{height} image into 8 octants")
        print(f"Base octant size: {octant_width}x{octant_height}")
        print(f"Expanded size with {overlap_percentage*100}% overlap: {expanded_width}x{expanded_height}")
    
    octant_files = []
    octant_regions = []
    
    # Define octant layout (4x2 grid)
    for row in range(2):
        for col in range(4):
            octant_id = row * 4 + col
            
            # Calculate overlapping region boundaries
            x_start = max(0, col * octant_width - overlap_x // 2)
            y_start = max(0, row * octant_height - overlap_y // 2)
            x_end = min(width, x_start + expanded_width)
            y_end = min(height, y_start + expanded_height)
            
            # Adjust for edge octants
            if col == 0:
                x_start = 0
            if col == 3:
                x_end = width
            if row == 0:
                y_start = 0
            if row == 1:
                y_end = height
            
            # Store region info
            region_info = {
                'octant_id': octant_id,
                'x_start': x_start,
                'y_start': y_start,
                'x_end': x_end,
                'y_end': y_end,
                'width': x_end - x_start,
                'height': y_end - y_start
            }
            octant_regions.append(region_info)
            
            # Crop the octant
            octant_img = full_img.crop((x_start, y_start, x_end, y_end))
            
            # Optionally upscale
            if upscale and target_resolution:
                octant_img = octant_img.resize(target_resolution, Image.Resampling.LANCZOS)
                if verbose and octant_id == 0:
                    print(f"Upscaling octants to {target_resolution[0]}x{target_resolution[1]}")
            
            # Save octant
            octant_filename = os.path.join(screenshots_folder, f"octant_{octant_id}.png")
            octant_img.save(octant_filename)
            octant_files.append(octant_filename)
    
    # Create mapping file for coordinate conversion
    mapping_data = {
        'full_resolution': {'width': width, 'height': height},
        'overlap_percentage': overlap_percentage,
        'octant_regions': octant_regions,
        'target_resolution': target_resolution if target_resolution else None
    }
    
    mapping_filename = os.path.join(screenshots_folder, "octant_mapping.json")
    with open(mapping_filename, 'w') as f:
        json.dump(mapping_data, f, indent=2)
    
    if verbose:
        print(f"Created {len(octant_files)} octant files")
        print(f"Mapping saved to: {mapping_filename}")
    
    return {
        'octant_files': octant_files,
        'octant_regions': octant_regions,
        'mapping_file': mapping_filename,
        'full_resolution': (width, height)
    }