#!/usr/bin/env python3
"""
Test script to validate that the application works without OpenCV.

This script simulates the cloud environment where OpenCV might not be available.
"""

import sys
import os
from pathlib import Path

# Temporarily hide cv2 to simulate cloud environment
import builtins
original_import = builtins.__import__

def mock_import(name, *args, **kwargs):
    if name == 'cv2':
        raise ImportError("No module named 'cv2'")
    return original_import(name, *args, **kwargs)

# Mock cv2 import
builtins.__import__ = mock_import

try:
    # Add app directory to path
    sys.path.append(str(Path(__file__).parent.parent / "app"))
    
    print("Testing imports without OpenCV...")
    
    # Test OCR service import
    from services.ocr_service import extract_text
    print("✅ OCR service imported successfully")
    
    # Test preprocess import
    from services.preprocess import preprocess_image
    print("✅ Preprocess service imported successfully")
    
    # Test layout segmentation import
    from services.layout_segmentation import segment_layout
    print("✅ Layout segmentation imported successfully")
    
    # Test math parser import
    from services.math_parser import parse_to_sympy
    print("✅ Math parser imported successfully")
    
    # Test grader import
    from services.grader import extract_and_grade
    print("✅ Grader imported successfully")
    
    print("\n🎉 All services imported successfully without OpenCV!")
    print("The application should work on Streamlit Cloud.")

except Exception as e:
    print(f"❌ Error importing services: {e}")
    sys.exit(1)

finally:
    # Restore original import
    builtins.__import__ = original_import
