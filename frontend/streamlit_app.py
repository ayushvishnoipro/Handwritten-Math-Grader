"""
Main Streamlit application for the handwritten math grader.

This module provides the main UI interface for uploading images,
extracting text via OCR, and grading mathematical solutions.
"""

import streamlit as st
import hashlib
import json
import os
from pathlib import Path
import sys

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not available, continue without it
    pass

# Add app directory to path for imports
sys.path.append(str(Path(__file__).parent.parent / "app"))

from components.uploader import render_file_uploader
from components.image_viewer import render_image_viewer
from components.ocr_editor import render_ocr_editor
from components.grader_ui import render_grader_ui
from services.grader import extract_and_grade
from services.exporter import save_docx


def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="Handwritten Math Grader",
        page_icon="📝",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📝 Handwritten Math Grader")
    st.markdown("Upload handwritten math solutions for automatic OCR extraction and grading.")
    
    # Initialize session state
    if 'ocr_results' not in st.session_state:
        st.session_state.ocr_results = None
    if 'grading_results' not in st.session_state:
        st.session_state.grading_results = None
    if 'edited_text' not in st.session_state:
        st.session_state.edited_text = ""
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Check API key availability from multiple sources
        has_gemini = _check_gemini_availability()
        
        st.write("**Available Services:**")
        st.write(f"🤖 Gemini: {'✅' if has_gemini else '❌'}")
        st.write(" Tesseract: ✅")
        
        if not has_gemini:
            st.warning("Gemini API key not found. Using Tesseract fallback.")
            with st.expander("ℹ️ API Key Setup"):
                st.markdown("""
                **Option 1: Environment Variables (Recommended for local development)**
                1. Copy `.env.example` to `.env`
                2. Add your Gemini API key: `GEMINI_API_KEY=your_key_here`
                
                **Option 2: Streamlit Secrets (For Streamlit Cloud)**
                1. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`
                2. Add your API key in the secrets file
                """)
        
        st.markdown("---")
        st.markdown("**Instructions:**")
        st.markdown("1. Upload question and solution images")
        st.markdown("2. Click 'Extract' to perform OCR")
        st.markdown("3. Review and edit extracted text")
        st.markdown("4. Click 'Grade' for evaluation")
        st.markdown("5. Export results to Word doc")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Files")
        question_file, solution_file = render_file_uploader()
        
        if question_file and solution_file:
            st.subheader("🖼️ Uploaded Images")
            render_image_viewer(question_file, solution_file)
    
    with col2:
        st.subheader("🔍 OCR Results")
        
        # Extract button
        if st.button("🚀 Extract Text", type="primary", disabled=not (question_file and solution_file)):
            if question_file and solution_file:
                with st.spinner("Extracting text from images..."):
                    # Cache OCR results by file hash
                    file_hash = hashlib.md5(
                        question_file.getvalue() + solution_file.getvalue()
                    ).hexdigest()
                    
                    @st.cache_data
                    def cached_extract_text(file_hash_key, q_bytes, s_bytes):
                        from services.ocr_service import extract_text
                        q_results = extract_text(q_bytes)
                        s_results = extract_text(s_bytes)
                        return {"question": q_results, "solution": s_results}
                    
                    ocr_results = cached_extract_text(
                        file_hash,
                        question_file.getvalue(),
                        solution_file.getvalue()
                    )
                    st.session_state.ocr_results = ocr_results
                    st.success("✅ Text extraction completed!")
        
        # Display OCR results
        if st.session_state.ocr_results:
            render_ocr_editor(st.session_state.ocr_results)
    
    # Grading section
    if st.session_state.ocr_results:
        st.markdown("---")
        st.subheader("🎯 Grading & Evaluation")
        
        col3, col4 = st.columns([1, 1])
        
        with col3:
            if st.button("🧮 Grade Solution", type="primary"):
                with st.spinner("Analyzing solution and grading..."):
                    # Use edited text if available, otherwise original OCR
                    text_to_grade = st.session_state.edited_text or ""
                    if not text_to_grade:
                        # Extract text from OCR results
                        solution_regions = st.session_state.ocr_results.get("solution", [])
                        text_to_grade = " ".join([r.get("text", "") for r in solution_regions])
                    
                    grading_results = extract_and_grade(
                        st.session_state.ocr_results,
                        text_to_grade
                    )
                    st.session_state.grading_results = grading_results
                    st.success("✅ Grading completed!")
        
        with col4:
            if st.session_state.grading_results and st.button("📄 Export to Word"):
                with st.spinner("Generating Word document..."):
                    # Prepare content for export
                    export_content = f"""
HANDWRITTEN MATH GRADER REPORT

OCR Results:
{json.dumps(st.session_state.ocr_results, indent=2)}

Grading Results:
{json.dumps(st.session_state.grading_results, indent=2)}

Edited Text:
{st.session_state.edited_text}
"""
                    
                    # Save to temporary file
                    output_path = "grading_report.docx"
                    save_docx(export_content, output_path)
                    
                    # Offer download
                    with open(output_path, "rb") as file:
                        st.download_button(
                            label="📥 Download Report",
                            data=file.read(),
                            file_name="grading_report.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        )
        
        # Display grading results
        if st.session_state.grading_results:
            render_grader_ui(st.session_state.grading_results)


def _check_gemini_availability() -> bool:
    """Check if Gemini API key is available from any source."""
    # Check Streamlit secrets
    try:
        if st.secrets.get("gemini", {}).get("api_key"):
            return True
    except Exception:
        pass
    
    # Check environment variable
    if os.getenv("GEMINI_API_KEY"):
        return True
    
    return False


if __name__ == "__main__":
    main()
