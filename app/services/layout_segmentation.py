"""
Layout segmentation service for identifying different regions in images.

Provides functionality to segment images into different regions like
text blocks, equations, diagrams, and figures.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from PIL import Image
from io import BytesIO
import uuid
from io import BytesIO

# Try to import OpenCV, fall back to PIL-only mode if not available
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


def segment_layout(image_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Segment image into different layout regions.
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        List of region dictionaries with bounding boxes and types
    """
    if not CV2_AVAILABLE:
        # Fallback to simple single-region layout when OpenCV is not available
        return _simple_layout_fallback(image_bytes)
    
    # Convert bytes to OpenCV image
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError("Could not decode image")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Find text regions
    text_regions = _find_text_regions(gray)
    
    # Find equation regions (areas with mathematical symbols)
    equation_regions = _find_equation_regions(gray)
    
    # Find diagram regions
    diagram_regions = _find_diagram_regions(gray)
    
    # Combine all regions
    all_regions = []
    all_regions.extend(text_regions)
    all_regions.extend(equation_regions)
    all_regions.extend(diagram_regions)
    
    # Remove overlapping regions and merge nearby ones
    merged_regions = _merge_overlapping_regions(all_regions)
    
    return merged_regions


def _find_text_regions(gray_image: np.ndarray) -> List[Dict[str, Any]]:
    """Find regions containing regular text."""
    regions = []
    
    # Apply morphological operations to find text areas
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
    morphed = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter by size and aspect ratio
        if w > 50 and h > 10 and w/h > 2:  # Text lines are typically wide
            region = {
                "region_id": str(uuid.uuid4()),
                "region_type": "text",
                "bbox": {"x": x, "y": y, "width": w, "height": h},
                "confidence": 0.8
            }
            regions.append(region)
    
    return regions


def _find_equation_regions(gray_image: np.ndarray) -> List[Dict[str, Any]]:
    """Find regions containing mathematical equations."""
    regions = []
    
    # Look for compact regions with mixed text and symbols
    # Equations often have different spacing patterns than regular text
    
    # Apply different morphological operations for equations
    kernel_eq = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morphed = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel_eq)
    
    # Find contours
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # Equations tend to be more square-like and contain dense content
        if w > 30 and h > 15 and 0.5 < w/h < 5:
            # Check if region contains potential mathematical symbols
            roi = gray_image[y:y+h, x:x+w]
            if _contains_math_symbols(roi):
                region = {
                    "region_id": str(uuid.uuid4()),
                    "region_type": "equation",
                    "bbox": {"x": x, "y": y, "width": w, "height": h},
                    "confidence": 0.7
                }
                regions.append(region)
    
    return regions


def _find_diagram_regions(gray_image: np.ndarray) -> List[Dict[str, Any]]:
    """Find regions containing diagrams or figures."""
    regions = []
    
    # Detect edges for geometric shapes
    edges = cv2.Canny(gray_image, 50, 150)
    
    # Find lines using HoughLines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
    
    if lines is not None:
        # Group nearby lines into potential diagram regions
        line_groups = _group_nearby_lines(lines)
        
        for group in line_groups:
            # Calculate bounding box for line group
            all_points = []
            for line in group:
                x1, y1, x2, y2 = line[0]
                all_points.extend([(x1, y1), (x2, y2)])
            
            if all_points:
                xs, ys = zip(*all_points)
                x, y = min(xs), min(ys)
                w, h = max(xs) - x, max(ys) - y
                
                # Filter by size
                if w > 50 and h > 50:
                    region = {
                        "region_id": str(uuid.uuid4()),
                        "region_type": "diagram",
                        "bbox": {"x": x, "y": y, "width": w, "height": h},
                        "confidence": 0.6
                    }
                    regions.append(region)
    
    return regions


def _contains_math_symbols(roi: np.ndarray) -> bool:
    """
    Check if region of interest contains mathematical symbols.
    
    This is a simple heuristic based on density and pattern analysis.
    """
    # Calculate pixel density
    height, width = roi.shape
    total_pixels = height * width
    dark_pixels = np.sum(roi < 128)
    density = dark_pixels / total_pixels
    
    # Math regions often have moderate density (not too sparse, not too dense)
    return 0.05 < density < 0.5


def _group_nearby_lines(lines: np.ndarray, distance_threshold: int = 50) -> List[List]:
    """Group lines that are close to each other."""
    if lines is None or len(lines) == 0:
        return []
    
    groups = []
    used = set()
    
    for i, line1 in enumerate(lines):
        if i in used:
            continue
            
        group = [line1]
        used.add(i)
        
        x1, y1, x2, y2 = line1[0]
        center1 = ((x1 + x2) // 2, (y1 + y2) // 2)
        
        for j, line2 in enumerate(lines):
            if j in used or i == j:
                continue
                
            x3, y3, x4, y4 = line2[0]
            center2 = ((x3 + x4) // 2, (y3 + y4) // 2)
            
            # Calculate distance between line centers
            distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
            
            if distance < distance_threshold:
                group.append(line2)
                used.add(j)
        
        if len(group) >= 2:  # Only keep groups with multiple lines
            groups.append(group)
    
    return groups


def _merge_overlapping_regions(regions: List[Dict[str, Any]], 
                              overlap_threshold: float = 0.3) -> List[Dict[str, Any]]:
    """Merge regions that overlap significantly."""
    if not regions:
        return []
    
    merged = []
    used = set()
    
    for i, region1 in enumerate(regions):
        if i in used:
            continue
            
        merged_region = region1.copy()
        used.add(i)
        
        bbox1 = region1["bbox"]
        
        for j, region2 in enumerate(regions):
            if j in used or i == j:
                continue
                
            bbox2 = region2["bbox"]
            
            # Calculate overlap
            overlap = _calculate_overlap(bbox1, bbox2)
            
            if overlap > overlap_threshold:
                # Merge the regions
                merged_bbox = _merge_bboxes(bbox1, bbox2)
                merged_region["bbox"] = merged_bbox
                merged_region["region_type"] = _determine_merged_type(
                    region1["region_type"], 
                    region2["region_type"]
                )
                used.add(j)
        
        merged.append(merged_region)
    
    return merged


def _calculate_overlap(bbox1: Dict[str, int], bbox2: Dict[str, int]) -> float:
    """Calculate overlap ratio between two bounding boxes."""
    x1, y1, w1, h1 = bbox1["x"], bbox1["y"], bbox1["width"], bbox1["height"]
    x2, y2, w2, h2 = bbox2["x"], bbox2["y"], bbox2["width"], bbox2["height"]
    
    # Calculate intersection
    left = max(x1, x2)
    top = max(y1, y2)
    right = min(x1 + w1, x2 + w2)
    bottom = min(y1 + h1, y2 + h2)
    
    if left < right and top < bottom:
        intersection = (right - left) * (bottom - top)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0
    
    return 0


def _merge_bboxes(bbox1: Dict[str, int], bbox2: Dict[str, int]) -> Dict[str, int]:
    """Merge two bounding boxes into one that contains both."""
    x1, y1, w1, h1 = bbox1["x"], bbox1["y"], bbox1["width"], bbox1["height"]
    x2, y2, w2, h2 = bbox2["x"], bbox2["y"], bbox2["width"], bbox2["height"]
    
    left = min(x1, x2)
    top = min(y1, y2)
    right = max(x1 + w1, x2 + w2)
    bottom = max(y1 + h1, y2 + h2)
    
    return {
        "x": left,
        "y": top,
        "width": right - left,
        "height": bottom - top
    }


def _determine_merged_type(type1: str, type2: str) -> str:
    """Determine the type for a merged region."""
    if type1 == type2:
        return type1
    
    # Priority order: equation > diagram > text
    priority = {"equation": 3, "diagram": 2, "text": 1}
    
    if priority.get(type1, 0) >= priority.get(type2, 0):
        return type1
    else:
        return type2


def _simple_layout_fallback(image_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Simple fallback layout segmentation when OpenCV is not available.
    
    Returns a single region covering the entire image.
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        List with single region covering entire image
    """
    try:
        # Use PIL to get image dimensions
        image = Image.open(BytesIO(image_bytes))
        width, height = image.size
        
        return [{
            "region_id": str(uuid.uuid4()),
            "region_type": "text",
            "bbox": {
                "x": 0,
                "y": 0,
                "width": width,
                "height": height
            },
            "confidence": 0.5  # Lower confidence since we're not doing real segmentation
        }]
    except Exception:
        # If even PIL fails, return a default region
        return [{
            "region_id": str(uuid.uuid4()),
            "region_type": "text", 
            "bbox": {
                "x": 0,
                "y": 0,
                "width": 800,  # Default dimensions
                "height": 600
            },
            "confidence": 0.3
        }]
