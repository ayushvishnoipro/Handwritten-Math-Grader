"""
Image viewer component for displaying uploaded images.

Provides a side-by-side view of question and solution images
with zoom and annotation capabilities.
"""

import streamlit as st
from PIL import Image
from io import BytesIO
from typing import Optional


def render_image_viewer(question_file: BytesIO, solution_file: BytesIO) -> None:
    """
    Render image viewer for question and solution images.
    
    Args:
        question_file: Uploaded question image file
        solution_file: Uploaded solution image file
    """
    if not question_file or not solution_file:
        return
    
    try:
        # Load images
        question_img = Image.open(question_file)
        solution_img = Image.open(solution_file)
        
        # Display images side by side
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📋 Question:**")
            st.image(
                question_img,
                caption=f"Question Image ({question_img.size[0]}x{question_img.size[1]})",
                width="stretch"
            )
            
            # Image info
            with st.expander("ℹ️ Image Details"):
                st.write(f"**Format:** {question_img.format}")
                st.write(f"**Mode:** {question_img.mode}")
                st.write(f"**Size:** {question_img.size}")
        
        with col2:
            st.markdown("**✍️ Solution:**")
            st.image(
                solution_img,
                caption=f"Solution Image ({solution_img.size[0]}x{solution_img.size[1]})",
                width="stretch"
            )
            
            # Image info
            with st.expander("ℹ️ Image Details"):
                st.write(f"**Format:** {solution_img.format}")
                st.write(f"**Mode:** {solution_img.mode}")
                st.write(f"**Size:** {solution_img.size}")
        
        # Reset file pointers for later use
        question_file.seek(0)
        solution_file.seek(0)
        
    except Exception as e:
        st.error(f"❌ Error loading images: {str(e)}")


def get_image_info(image: Image.Image) -> dict:
    """
    Extract metadata and information from PIL Image.
    
    Args:
        image: PIL Image object
        
    Returns:
        Dictionary containing image information
    """
    return {
        "format": image.format,
        "mode": image.mode,
        "size": image.size,
        "width": image.size[0],
        "height": image.size[1],
        "has_transparency": image.mode in ("RGBA", "LA", "P")
    }
