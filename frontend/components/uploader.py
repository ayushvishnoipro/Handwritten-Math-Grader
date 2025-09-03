"""
File uploader component for question and solution images.

Provides a clean interface for uploading image files with validation
and preview capabilities.
"""

import streamlit as st
from typing import Tuple, Optional
from io import BytesIO


def render_file_uploader() -> Tuple[Optional[BytesIO], Optional[BytesIO]]:
    """
    Render file upload widgets for question and solution images.
    
    Returns:
        Tuple of (question_file, solution_file) or (None, None) if not uploaded
    """
    st.markdown("Upload both the question and solution images to begin.")
    
    # Question file uploader
    question_file = st.file_uploader(
        "📋 Question Image",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        help="Upload the question/problem statement image",
        key="question_uploader"
    )
    
    # Solution file uploader
    solution_file = st.file_uploader(
        "✍️ Solution Image",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        help="Upload the handwritten solution image",
        key="solution_uploader"
    )
    
    # Validation feedback
    if question_file and solution_file:
        st.success("✅ Both files uploaded successfully!")
        
        # Display file info
        col1, col2 = st.columns(2)
        with col1:
            st.caption(f"Question: {question_file.name} ({question_file.size} bytes)")
        with col2:
            st.caption(f"Solution: {solution_file.name} ({solution_file.size} bytes)")
            
    elif question_file or solution_file:
        st.warning("⚠️ Please upload both question and solution images.")
    
    return question_file, solution_file


def validate_image_file(file: BytesIO) -> bool:
    """
    Validate that uploaded file is a valid image.
    
    Args:
        file: Uploaded file object
        
    Returns:
        True if valid image, False otherwise
    """
    if not file:
        return False
        
    try:
        from PIL import Image
        Image.open(file)
        file.seek(0)  # Reset file pointer
        return True
    except Exception:
        return False
