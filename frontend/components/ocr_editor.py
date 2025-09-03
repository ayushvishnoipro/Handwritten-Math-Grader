"""
OCR editor component for viewing and editing extracted text.

Provides an interface to view OCR results and make manual corrections
before grading.
"""

import streamlit as st
import json
import re
from typing import Dict, List, Any


def render_ocr_editor(ocr_results: Dict[str, List[Dict[str, Any]]]) -> None:
    """
    Render OCR results editor interface.
    
    Args:
        ocr_results: Dictionary containing OCR results for question and solution
    """
    if not ocr_results:
        st.info("No OCR results to display. Please extract text first.")
        return
    
    # Display tabs for question and solution
    tab1, tab2 = st.tabs(["📋 Question OCR", "✍️ Solution OCR"])
    
    with tab1:
        _render_ocr_regions("question", ocr_results.get("question", []))
    
    with tab2:
        _render_ocr_regions("solution", ocr_results.get("solution", []))
    
    # Combined editable text area
    st.markdown("---")
    st.subheader("✏️ Edit Extracted Text")
    
    # Combine all text for editing
    all_text = _extract_combined_text(ocr_results)
    
    edited_text = st.text_area(
        "Edit the extracted text if needed:",
        value=all_text,
        height=150,
        help="Make corrections to the OCR text before grading"
    )
    
    # Update session state with edited text
    st.session_state.edited_text = edited_text
    
    if edited_text != all_text:
        st.info("✏️ Text has been modified from original OCR output.")


def _render_ocr_regions(label: str, regions: List[Dict[str, Any]]) -> None:
    """
    Render OCR regions for a specific image.
    
    Args:
        label: Label for the regions (question/solution)
        regions: List of OCR region dictionaries
    """
    if not regions:
        st.warning(f"No {label} regions detected.")
        return
    
    st.write(f"**Detected {len(regions)} region(s):**")
    
    for i, region in enumerate(regions):
        with st.expander(f"Region {i+1} - {region.get('region_type', 'unknown')}"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write("**Extracted Text:**")
                # Display text with proper formatting
                text_content = region.get('text', 'No text')
                if '\n' in text_content:
                    # Multi-line content - display with line breaks preserved
                    st.text(text_content)
                else:
                    # Single line content
                    st.code(text_content, language=None)
                
                if region.get('latex'):
                    st.write("**LaTeX:**")
                    st.code(region.get('latex', ''), language='latex')
            
            with col2:
                st.write("**Details:**")
                st.write(f"ID: `{region.get('region_id', 'N/A')}`")
                st.write(f"Type: `{region.get('region_type', 'N/A')}`")
                st.write(f"Confidence: `{region.get('confidence', 0):.2f}`")
                
                bbox = region.get('bbox', {})
                if bbox:
                    st.write(f"BBox: `{bbox}`")


def _extract_combined_text(ocr_results: Dict[str, List[Dict[str, Any]]]) -> str:
    """
    Extract and combine all text from OCR results with proper formatting.
    
    Args:
        ocr_results: Dictionary containing OCR results
        
    Returns:
        Combined text string with proper line structure
    """
    text_parts = []
    
    # Add question text
    question_regions = ocr_results.get("question", [])
    if question_regions:
        text_parts.append("=== QUESTION ===")
        for region in question_regions:
            if region.get('text'):
                # Preserve line structure from OCR
                region_text = region['text'].strip()
                if region_text:
                    text_parts.append(region_text)
                    text_parts.append("")  # Add blank line for separation
    
    # Add solution text
    solution_regions = ocr_results.get("solution", [])
    if solution_regions:
        if text_parts:  # Add separation if there was question text
            text_parts.append("")
        text_parts.append("=== SOLUTION ===")
        for region in solution_regions:
            if region.get('text'):
                # Preserve line structure from OCR
                region_text = region['text'].strip()
                if region_text:
                    text_parts.append(region_text)
                    text_parts.append("")  # Add blank line for separation
    
    # Clean up the result
    result = "\n".join(text_parts)
    
    # Remove excessive blank lines
    result = re.sub(r'\n{3,}', '\n\n', result)
    
    return result.strip()


def display_ocr_json(ocr_results: Dict[str, List[Dict[str, Any]]]) -> None:
    """
    Display raw OCR results as formatted JSON.
    
    Args:
        ocr_results: Dictionary containing OCR results
    """
    with st.expander("🔍 View Raw OCR Data"):
        st.json(ocr_results)
