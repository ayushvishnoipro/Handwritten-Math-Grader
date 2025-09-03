"""
OCR service for extracting text from images using Gemini and Tesseract.

Provides a unified interface for text extraction using Gemini as primary
and Tesseract as fallback when Gemini is unavailable.
"""

import streamlit as st
import os
from typing import List, Dict, Any, Optional
import cv2
import numpy as np
from PIL import Image
import pytesseract
from io import BytesIO
import uuid
import re

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not available, continue without it
    pass

from .gemini_client import get_gemini_client
from .preprocess import preprocess_image


def extract_text(image_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Extract text from image using available OCR engines.
    
    Tries engines in order: Gemini -> Tesseract
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        List of region dictionaries with structure:
        {
            "region_id": str,
            "region_type": str,
            "bbox": dict,
            "text": str,
            "latex": str,
            "confidence": float
        }
    """
    regions = []
    
    # Try Gemini first
    try:
        gemini_regions = _extract_with_gemini(image_bytes)
        if gemini_regions:
            return gemini_regions
    except Exception as e:
        st.warning(f"Gemini OCR failed: {str(e)}")
    
    # Fallback to Tesseract
    try:
        tesseract_regions = _extract_with_tesseract(image_bytes)
        return tesseract_regions
    except Exception as e:
        st.error(f"All OCR engines failed. Tesseract error: {str(e)}")
        return []


def _extract_with_gemini(image_bytes: bytes) -> List[Dict[str, Any]]:
    """Extract text using Gemini vision API."""
    client = get_gemini_client()
    if not client:
        raise RuntimeError("Gemini client not available")
    
    try:
        # First try structured mathematical content extraction
        structured_content = client.extract_mathematical_content(image_bytes)
        
        if structured_content and structured_content.get("raw_text"):
            # Create regions from structured content
            regions = []
            
            # Main text region
            main_text = structured_content.get("raw_text", "")
            if main_text.strip():
                # Structure and clean the extracted text
                structured_text = structure_mathematical_text(main_text)
                
                region = {
                    "region_id": str(uuid.uuid4()),
                    "region_type": "text",
                    "bbox": {"x": 0, "y": 0, "width": 0, "height": 0},
                    "text": structured_text,
                    "latex": "",
                    "confidence": 0.9,
                    "structured_content": structured_content
                }
                regions.append(region)
            
            # Add separate regions for equations if available
            equations = structured_content.get("equations", [])
            for i, equation in enumerate(equations):
                if equation and equation.strip():
                    eq_region = {
                        "region_id": str(uuid.uuid4()),
                        "region_type": "equation",
                        "bbox": {"x": 0, "y": 0, "width": 0, "height": 0},
                        "text": equation.strip(),
                        "latex": "",
                        "confidence": 0.85
                    }
                    regions.append(eq_region)
            
            return regions
            
    except Exception as e:
        # Fallback to simple text extraction
        pass
    
    # Simple text extraction fallback
    try:
        extracted_text = client.extract_text_from_image(image_bytes)
        
        # Preserve original formatting and structure
        formatted_text = _preserve_original_formatting(extracted_text)
        
        # Detect and structure mathematical sections
        sections = _detect_mathematical_sections(formatted_text)
        
        regions = []
        
        # Create regions for each section
        for section in sections:
            formatted_content = _format_mathematical_content(section['content'])
            
            region = {
                "region_id": str(uuid.uuid4()),
                "region_type": section['type'],
                "bbox": {"x": 0, "y": 0, "width": 0, "height": 0},
                "text": formatted_content,
                "latex": "",
                "confidence": 0.9
            }
            regions.append(region)
        
        # If no sections detected, return as single region
        if not regions:
            region = {
                "region_id": str(uuid.uuid4()),
                "region_type": "text",
                "bbox": {"x": 0, "y": 0, "width": 0, "height": 0},
                "text": formatted_text,
                "latex": "",
                "confidence": 0.9
            }
            regions = [region]
        
        return regions
        
    except Exception as e:
        raise RuntimeError(f"Gemini text extraction failed: {str(e)}")


def _extract_with_tesseract(image_bytes: bytes) -> List[Dict[str, Any]]:
    """Extract text using Tesseract OCR with layout preservation."""
    # Preprocess image
    processed_image = preprocess_image(image_bytes)
    
    # Convert to PIL Image for Tesseract
    pil_image = Image.fromarray(processed_image)
    
    # Extract text with layout preservation
    try:
        # First try to get text with layout information
        layout_text = pytesseract.image_to_string(
            pil_image, 
            config='--psm 6'  # Uniform block of text
        )
        
        if layout_text.strip():
            # Preserve formatting and structure
            formatted_text = _preserve_original_formatting(layout_text)
            
            # Detect mathematical sections
            sections = _detect_mathematical_sections(formatted_text)
            
            regions = []
            
            # Create regions for each section
            for section in sections:
                formatted_content = _format_mathematical_content(section['content'])
                
                region = {
                    "region_id": str(uuid.uuid4()),
                    "region_type": section['type'],
                    "bbox": {"x": 0, "y": 0, "width": 0, "height": 0},
                    "text": formatted_content,
                    "latex": "",
                    "confidence": 0.7
                }
                regions.append(region)
            
            # If no sections detected, return as single region
            if not regions:
                region = {
                    "region_id": str(uuid.uuid4()),
                    "region_type": "text",
                    "bbox": {"x": 0, "y": 0, "width": 0, "height": 0},
                    "text": formatted_text,
                    "latex": "",
                    "confidence": 0.7
                }
                regions = [region]
            
            return regions
        
    except Exception:
        pass
    
    # Fallback to detailed extraction with bounding boxes
    try:
        # Get detailed data with bounding boxes
        data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
        
        # Group text by lines based on y-coordinates
        line_groups = {}
        
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            conf = int(data['conf'][i])
            
            if text and conf > 30:  # Filter low confidence detections
                y = data['top'][i]
                x = data['left'][i]
                
                # Group by approximate line (within 10 pixels)
                line_key = None
                for existing_y in line_groups.keys():
                    if abs(y - existing_y) <= 10:
                        line_key = existing_y
                        break
                
                if line_key is None:
                    line_key = y
                    line_groups[line_key] = []
                
                line_groups[line_key].append((x, text))
        
        # Sort lines by y-coordinate and reconstruct text with line breaks
        sorted_lines = sorted(line_groups.keys())
        reconstructed_lines = []
        
        for y in sorted_lines:
            # Sort words in line by x-coordinate
            words = sorted(line_groups[y], key=lambda item: item[0])
            line_text = ' '.join([word[1] for word in words])
            reconstructed_lines.append(line_text)
        
        # Join lines and format
        combined_text = '\n'.join(reconstructed_lines)
        formatted_text = _preserve_original_formatting(combined_text)
        
        region = {
            "region_id": str(uuid.uuid4()),
            "region_type": "text",
            "bbox": {"x": 0, "y": 0, "width": 0, "height": 0},
            "text": formatted_text,
            "latex": "",
            "confidence": 0.7
        }
        
        return [region]
        
    except Exception as e:
        # Final fallback to simple text extraction
        try:
            simple_text = pytesseract.image_to_string(pil_image)
            if simple_text.strip():
                formatted_text = _preserve_original_formatting(simple_text)
                
                region = {
                    "region_id": str(uuid.uuid4()),
                    "region_type": "text",
                    "bbox": {"x": 0, "y": 0, "width": 0, "height": 0},
                    "text": formatted_text,
                    "latex": "",
                    "confidence": 0.5
                }
                return [region]
        except:
            pass
        
        raise e


def _merge_bboxes(bbox1: Dict[str, int], bbox2: Dict[str, int]) -> Dict[str, int]:
    """Merge two bounding boxes into one that contains both."""
    x1, y1 = bbox1['x'], bbox1['y']
    x2, y2 = bbox1['x'] + bbox1['width'], bbox1['y'] + bbox1['height']
    
    x3, y3 = bbox2['x'], bbox2['y']
    x4, y4 = bbox2['x'] + bbox2['width'], bbox2['y'] + bbox2['height']
    
    # Find the merged bounding box
    merged_x = min(x1, x3)
    merged_y = min(y1, y3)
    merged_width = max(x2, x4) - merged_x
    merged_height = max(y2, y4) - merged_y
    
    return {
        'x': merged_x,
        'y': merged_y,
        'width': merged_width,
        'height': merged_height
    }


def _clean_ocr_text(text: str) -> str:
    """
    Clean OCR text and convert LaTeX expressions to normal mathematical notation.
    
    Args:
        text: Raw OCR text
        
    Returns:
        Cleaned and normalized text with proper line formatting
    """
    if not text:
        return ""
    
    # Convert LaTeX expressions to normal notation
    cleaned = _convert_latex_expressions(text)
    
    # Split into lines for better processing
    lines = cleaned.split('\n')
    processed_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Skip section headers
        if re.match(r'^={2,}.*={2,}$', line):
            continue
            
        # Clean up excessive whitespace within the line
        line = re.sub(r'\s+', ' ', line)
        
        # Format mathematical expressions properly
        line = _format_mathematical_line(line)
        
        if line:  # Only add non-empty lines
            processed_lines.append(line)
    
    return '\n'.join(processed_lines)


def _convert_latex_expressions(text: str) -> str:
    """Convert LaTeX mathematical expressions to readable format."""
    
    # Remove dollar signs and LaTeX environment markers
    text = re.sub(r'\$([^$]*)\$', r'\1', text)
    text = re.sub(r'\\begin\{[^}]+\}.*?\\end\{[^}]+\}', '', text, flags=re.DOTALL)
    
    # Convert common LaTeX commands
    conversions = {
        r'\\int': 'integral of',
        r'\\frac\{([^}]+)\}\{([^}]+)\}': r'(\1)/(\2)',
        r'\\dfrac\{([^}]+)\}\{([^}]+)\}': r'(\1)/(\2)',
        r'\\cfrac\{([^}]+)\}\{([^}]+)\}': r'(\1)/(\2)',
        r'\\tan': 'tan',
        r'\\sec': 'sec', 
        r'\\csc': 'csc',
        r'\\cot': 'cot',
        r'\\sin': 'sin',
        r'\\cos': 'cos',
        r'\\log': 'log',
        r'\\ln': 'ln',
        r'\\exp': 'exp',
        r'\\arctan': 'arctan',
        r'\\arcsin': 'arcsin',
        r'\\arccos': 'arccos',
        r'\\sqrt\{([^}]+)\}': r'sqrt(\1)',
        r'\\sqrt\[(\d+)\]\{([^}]+)\}': r'(\2)^(1/\1)',  # nth root
        r'\\Rightarrow': ' => ',
        r'\\rightarrow': ' -> ',
        r'\\leftarrow': ' <- ',
        r'\\leftrightarrow': ' <-> ',
        r'\\,': ' ',
        r'\\;': ' ',
        r'\\:': ' ',
        r'\\!': '',
        r'\\quad': ' ',
        r'\\qquad': '  ',
        r'\\\\': '\n',
        r'\\text\{([^}]+)\}': r'\1',
        r'\\mathrm\{([^}]+)\}': r'\1',
        r'\\mathbf\{([^}]+)\}': r'\1',
        r'\\mathit\{([^}]+)\}': r'\1',
        r'\\alpha': 'alpha',
        r'\\beta': 'beta',
        r'\\gamma': 'gamma',
        r'\\delta': 'delta',
        r'\\theta': 'theta',
        r'\\pi': 'pi',
        r'\\phi': 'phi',
        r'\\omega': 'omega',
        r'\\lambda': 'lambda',
        r'\\mu': 'mu',
        r'\\sigma': 'sigma',
        r'\\tau': 'tau',
        r'\\sum': 'sum of',
        r'\\prod': 'product of',
        r'\\lim': 'limit of',
        r'\\infty': 'infinity',
        r'\\cdot': '*',
        r'\\times': '*',
        r'\\div': '/',
        r'\\pm': '+/-',
        r'\\mp': '-/+',
        r'\\leq': '<=',
        r'\\geq': '>=',
        r'\\neq': '!=',
        r'\\approx': '≈',
        r'\\equiv': '≡',
        r'\\left': '',
        r'\\right': '',
    }
    
    for pattern, replacement in conversions.items():
        text = re.sub(pattern, replacement, text)
    
    # Handle exponents and subscripts
    text = re.sub(r'\^(\d+)', r'^(\1)', text)
    text = re.sub(r'\^\{([^}]+)\}', r'^(\1)', text)
    text = re.sub(r'\_(\d+)', r'_(\1)', text)
    text = re.sub(r'\_\{([^}]+)\}', r'_(\1)', text)
    
    # Clean up braces and remaining LaTeX commands
    text = re.sub(r'[{}]', '', text)
    text = re.sub(r'\\[a-zA-Z]+', '', text)  # Remove any remaining LaTeX commands
    
    # Format integrals nicely
    text = re.sub(r'integral of\s*([^d]+)\s*d([a-zA-Z])', r'integral of \1 with respect to \2', text)
    
    # Clean up spacing
    text = re.sub(r'\s+', ' ', text)
    
    return text


def _format_mathematical_line(line: str) -> str:
    """
    Format a single line of mathematical text for better readability.
    
    Args:
        line: A line of text that may contain mathematical expressions
        
    Returns:
        Formatted line with proper mathematical notation
    """
    if not line or not line.strip():
        return ""
    
    # Remove leading/trailing whitespace
    line = line.strip()
    
    # Skip lines that are just formatting markers
    skip_patterns = [
        r'^\*+$',  # Lines with just asterisks
        r'^=+$',   # Lines with just equals signs
        r'^-+$',   # Lines with just dashes
        r'^\d+\*\*$',  # Lines like "10**"
    ]
    
    for pattern in skip_patterns:
        if re.match(pattern, line):
            return ""
    
    # Format problem/question numbers
    if re.match(r'^\*\*(\d+)\*\*$', line):
        match = re.match(r'^\*\*(\d+)\*\*$', line)
        return f"Problem {match.group(1)}:"
    
    # Format arrows and implications
    line = re.sub(r'=>', '→', line)
    line = re.sub(r'->', '→', line)
    
    # Ensure proper spacing around mathematical operators
    line = re.sub(r'([a-zA-Z0-9])\+', r'\1 + ', line)
    line = re.sub(r'\+([a-zA-Z0-9])', r'+ \1', line)
    line = re.sub(r'([a-zA-Z0-9])-', r'\1 - ', line)
    line = re.sub(r'-([a-zA-Z0-9])', r'- \1', line)
    line = re.sub(r'([a-zA-Z0-9])=', r'\1 = ', line)
    line = re.sub(r'=([a-zA-Z0-9])', r'= \1', line)
    
    # Clean up multiple spaces
    line = re.sub(r'\s+', ' ', line)
    
    return line.strip()


def structure_mathematical_text(text: str) -> str:
    """
    Structure mathematical text into a clean, line-by-line format.
    
    Args:
        text: Raw mathematical text
        
    Returns:
        Structured text with proper formatting
    """
    if not text or not text.strip():
        return ""
    
    lines = text.split('\n')
    structured_lines = []
    current_problem = None
    
    for line in lines:
        line = line.strip()
        
        if not line:
            # Add blank line if we're between problems
            if structured_lines and not structured_lines[-1] == "":
                structured_lines.append("")
            continue
        
        # Check if this is a problem number
        problem_match = re.match(r'^\*\*(\d+)\*\*', line)
        if problem_match:
            if structured_lines and not structured_lines[-1] == "":
                structured_lines.append("")  # Add separation
            structured_lines.append(f"Problem {problem_match.group(1)}:")
            current_problem = problem_match.group(1)
            continue
        
        # Clean and format the line
        formatted_line = _format_mathematical_line(line)
        if formatted_line:
            # Add proper indentation for solution steps
            if current_problem and not formatted_line.startswith("Problem"):
                if any(op in formatted_line for op in ['=', '→', 'integral', 'let', 'therefore']):
                    formatted_line = "  " + formatted_line  # Indent solution steps            
            structured_lines.append(formatted_line)
    
    # Clean up multiple consecutive blank lines
    result_lines = []
    prev_blank = False
    
    for line in structured_lines:
        if line == "":
            if not prev_blank:
                result_lines.append(line)
            prev_blank = True
        else:
            result_lines.append(line)
            prev_blank = False
    
    return '\n'.join(result_lines).strip()


def _preserve_original_formatting(text: str) -> str:
    """
    Preserve the original formatting and layout from OCR text.
    
    Args:
        text: Raw OCR text
        
    Returns:
        Text with preserved formatting
    """
    if not text:
        return ""
    
    # Clean up excessive whitespace while preserving structure
    lines = text.split('\n')
    formatted_lines = []
    
    for line in lines:
        # Preserve leading whitespace for indentation
        stripped = line.rstrip()
        if stripped:  # Non-empty line
            # Convert mathematical notation while preserving structure
            formatted_line = _convert_latex_expressions(stripped)
            formatted_lines.append(formatted_line)
        else:  # Empty line - preserve as spacing
            formatted_lines.append('')
    
    # Join lines back together
    result = '\n'.join(formatted_lines)
    
    # Clean up excessive blank lines while preserving intentional spacing
    result = re.sub(r'\n{4,}', '\n\n\n', result)  # Max 3 blank lines
    
    return result.strip()


def _detect_mathematical_sections(text: str) -> List[Dict[str, Any]]:
    """
    Detect different mathematical sections in the text.
    
    Args:
        text: Input text
        
    Returns:
        List of section dictionaries with type and content
    """
    sections = []
    lines = text.split('\n')
    current_section = None
    current_lines = []
    
    for line in lines:
        line = line.strip()
        
        if not line:  # Empty line
            if current_lines:
                current_lines.append('')
            continue
            
        # Detect section headers
        if re.match(r'^\*\*\d+\*\*$', line):  # **10**, **12**, etc.
            # Save previous section
            if current_section and current_lines:
                sections.append({
                    'type': current_section,
                    'content': '\n'.join(current_lines).strip()
                })
            
            # Start new problem section
            current_section = 'problem'
            current_lines = [line]
            
        elif line.startswith('=== '):  # Section markers
            # Save previous section
            if current_section and current_lines:
                sections.append({
                    'type': current_section,
                    'content': '\n'.join(current_lines).strip()
                })
            
            # Determine section type
            if 'QUESTION' in line.upper():
                current_section = 'question'
            elif 'SOLUTION' in line.upper():
                current_section = 'solution'
            else:
                current_section = 'section'
            
            current_lines = [line]
            
        else:
            # Add to current section
            if current_section is None:
                current_section = 'content'
            current_lines.append(line)
    
    # Add final section
    if current_section and current_lines:
        sections.append({
            'type': current_section,
            'content': '\n'.join(current_lines).strip()
        })
    
    return sections


def _format_mathematical_content(content: str) -> str:
    """
    Format mathematical content to be more readable while preserving structure.
    
    Args:
        content: Mathematical content
        
    Returns:
        Formatted content
    """
    # Split into lines for processing
    lines = content.split('\n')
    formatted_lines = []
    
    for line in lines:
        if not line.strip():
            formatted_lines.append('')
            continue
            
        # Process mathematical expressions
        formatted_line = line
        
        # Format arrows and logical flow
        formatted_line = re.sub(r'\\Rightarrow', ' ⇒ ', formatted_line)
        formatted_line = re.sub(r'=>', ' ⇒ ', formatted_line)
        formatted_line = re.sub(r'->', ' → ', formatted_line)
        
        # Format equals signs with proper spacing
        formatted_line = re.sub(r'\s*=\s*', ' = ', formatted_line)
        
        # Format operators with spacing
        formatted_line = re.sub(r'\s*\+\s*', ' + ', formatted_line)
        formatted_line = re.sub(r'\s*-\s*', ' - ', formatted_line)
        formatted_line = re.sub(r'\s*\*\s*', ' * ', formatted_line)
        
        # Clean up multiple spaces
        formatted_line = re.sub(r'\s+', ' ', formatted_line)
        
        formatted_lines.append(formatted_line)
    
    return '\n'.join(formatted_lines)


@st.cache_data
def cached_extract_text(file_hash: str, image_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Cached version of text extraction.
    
    Args:
        file_hash: Hash of the file for caching
        image_bytes: Raw image bytes
        
    Returns:
        List of extracted regions
    """
    return extract_text(image_bytes)
