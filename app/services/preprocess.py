"""
Image preprocessing utilities for improving OCR accuracy.

Provides functions to enhance image quality, remove noise,
and optimize images for text extraction.
"""

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from io import BytesIO
from typing import Optional

# Try to import cv2, but make it optional for cloud deployment
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


def preprocess_image(image_bytes: bytes, enhance_contrast: bool = True, 
                    denoise: bool = True, resize_factor: Optional[float] = None) -> np.ndarray:
    """
    Preprocess image to improve OCR accuracy.
    
    Args:
        image_bytes: Raw image bytes
        enhance_contrast: Whether to enhance contrast
        denoise: Whether to apply denoising
        resize_factor: Factor to resize image (None for no resize)
        
    Returns:
        Preprocessed image as numpy array
    """
    if CV2_AVAILABLE:
        return _preprocess_with_opencv(image_bytes, enhance_contrast, denoise, resize_factor)
    else:
        return _preprocess_with_pil(image_bytes, enhance_contrast, denoise, resize_factor)


def _preprocess_with_opencv(image_bytes: bytes, enhance_contrast: bool = True, 
                           denoise: bool = True, resize_factor: Optional[float] = None) -> np.ndarray:
    """Preprocess image using OpenCV (when available)."""
    # Convert bytes to OpenCV image
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError("Could not decode image")
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Resize if requested
    if resize_factor and resize_factor != 1.0:
        height, width = gray.shape[:2]
        new_height = int(height * resize_factor)
        new_width = int(width * resize_factor)
        gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # Enhance contrast
    if enhance_contrast:
        gray = enhance_image_contrast(gray)
    
    # Denoise
    if denoise:
        gray = cv2.fastNlMeansDenoising(gray, h=10)
    
    # Apply adaptive thresholding for better text detection
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    return binary


def enhance_image_contrast(image: np.ndarray) -> np.ndarray:
    """
    Enhance image contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    Args:
        image: Grayscale image as numpy array
        
    Returns:
        Contrast-enhanced image
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)


def remove_background_noise(image: np.ndarray) -> np.ndarray:
    """
    Remove background noise and improve text clarity.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Cleaned image
    """
    # Apply morphological operations to remove noise
    kernel = np.ones((1, 1), np.uint8)
    
    # Opening to remove noise
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Closing to fill gaps
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return closing


def detect_text_orientation(image: np.ndarray) -> float:
    """
    Detect text orientation angle in the image.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Rotation angle in degrees
    """
    # Use HoughLines to detect text lines and estimate orientation
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
    
    if lines is not None:
        angles = []
        for line in lines:
            rho, theta = line[0]
            angle = theta * 180 / np.pi
            # Normalize angle to [-90, 90] range
            if angle > 90:
                angle -= 180
            angles.append(angle)
        
        # Return the median angle
        if angles:
            return float(np.median(angles))
    
    return 0.0


def correct_skew(image: np.ndarray, angle: Optional[float] = None) -> np.ndarray:
    """
    Correct image skew based on detected or provided angle.
    
    Args:
        image: Input image as numpy array
        angle: Rotation angle in degrees (auto-detect if None)
        
    Returns:
        Skew-corrected image
    """
    if angle is None:
        angle = detect_text_orientation(image)
    
    if abs(angle) < 0.5:  # Skip correction for very small angles
        return image
    
    # Get image dimensions
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Create rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Apply rotation
    rotated = cv2.warpAffine(image, rotation_matrix, (w, h), 
                           flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return rotated


def segment_lines(image: np.ndarray) -> list:
    """
    Segment image into individual text lines.
    
    Args:
        image: Binary image as numpy array
        
    Returns:
        List of line images
    """
    # Find horizontal projection
    horizontal_projection = np.sum(image == 0, axis=1)
    
    # Find line boundaries
    lines = []
    in_line = False
    start_y = 0
    
    for i, count in enumerate(horizontal_projection):
        if count > 0 and not in_line:
            # Start of a line
            start_y = i
            in_line = True
        elif count == 0 and in_line:
            # End of a line
            if i - start_y > 5:  # Minimum line height
                line_image = image[start_y:i, :]
                lines.append(line_image)
            in_line = False
    
    # Handle case where last line extends to bottom
    if in_line and len(horizontal_projection) - start_y > 5:
        line_image = image[start_y:, :]
        lines.append(line_image)
    
    return lines
