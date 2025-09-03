"""
Google Gemini client for text generation and image analysis.

Provides a wrapper around the Google Generative AI API with
Streamlit secrets integration and environment variable fallback.
"""

import streamlit as st
import os
from typing import Optional, Dict, Any
import base64
from io import BytesIO
from PIL import Image

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not available, continue without it
    pass


class GeminiClient:
    """Client for Google Gemini API operations."""
    
    def __init__(self):
        """Initialize Gemini client with API key from secrets or environment."""
        try:
            import google.generativeai as genai
            self.genai = genai
            
            # Try to get API key from multiple sources
            api_key = self._get_api_key()
            if not api_key:
                raise ValueError("Gemini API key not found in secrets or environment variables")
            
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            self.available = True
            
        except Exception as e:
            self.available = False
            self.error_message = str(e)
    
    def _get_api_key(self) -> Optional[str]:
        """Get API key from Streamlit secrets or environment variables."""
        # Try Streamlit secrets first (for Streamlit Cloud deployment)
        try:
            api_key = st.secrets.get("gemini", {}).get("api_key")
            if api_key:
                return api_key
        except Exception:
            pass
        
        # Try environment variable
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            return api_key
        
        return None
    
    def is_available(self) -> bool:
        """Check if Gemini client is available."""
        return self.available
    
    def generate_text(self, prompt: str, **kwargs) -> str:
        """
        Generate text using Gemini model.
        
        Args:
            prompt: Text prompt for generation
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
            
        Raises:
            RuntimeError: If client is not available
        """
        if not self.available:
            raise RuntimeError(f"Gemini client not available: {self.error_message}")
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            raise RuntimeError(f"Text generation failed: {str(e)}")
    
    def extract_text_from_image(self, image_bytes: bytes, prompt: Optional[str] = None) -> str:
        """
        Extract text from image using Gemini vision capabilities.
        
        Args:
            image_bytes: Raw image bytes
            prompt: Optional custom prompt for extraction
            
        Returns:
            Extracted text from image
            
        Raises:
            RuntimeError: If client is not available or extraction fails
        """
        if not self.available:
            raise RuntimeError(f"Gemini client not available: {self.error_message}")
        
        try:
            # Convert bytes to PIL Image
            image = Image.open(BytesIO(image_bytes))
            
            # Enhanced prompt for mathematical OCR with layout preservation
            if prompt is None:
                prompt = """
                Extract all text from this handwritten mathematical image while preserving the exact layout and formatting.
                
                LAYOUT PRESERVATION REQUIREMENTS:
                1. Maintain the exact line breaks and vertical spacing as they appear in the image
                2. Preserve indentation and alignment of mathematical expressions
                3. Keep numbered problems, sections, or steps in their original format  
                4. Preserve the sequential order of mathematical steps exactly as written
                5. Maintain any arrows (→, ⇒) or symbols that show logical flow between steps
                6. Keep equations, fractions, and expressions positioned as they appear
                7. Preserve any bullet points, numbering, or section headers exactly
                8. Use line breaks to separate expressions only where they appear in the image
                9. Maintain the visual hierarchy and structure of the content
                10. If content appears in columns, preserve the column structure
                
                MATHEMATICAL NOTATION CONVERSION:
                - Convert symbols to readable text: ∫→integral, √→sqrt(), ∑→sum, ∏→product
                - For fractions: use (numerator)/(denominator) format
                - For exponents: use ^ notation (e.g., x^2, e^x)
                - For subscripts: use _ notation (e.g., x_1, a_n)
                - Trigonometric functions: sin, cos, tan, sec, csc, cot, arcsin, arccos, arctan
                - Logarithms: log, ln (preserve as written)
                - Preserve parentheses, brackets, and braces exactly as shown
                - Convert mathematical operators: × → *, ÷ → /, ± → +/-, ≈ → ~, ≠ → !=
                
                OUTPUT FORMAT:
                Reproduce the text exactly as it appears in the image with:
                - Original line breaks and spacing
                - Same indentation and alignment
                - Same numbering and sectioning
                - Same mathematical flow and structure
                
                Do not add explanations, reorganize content, or change the visual structure.
                Extract only what is written, preserving its exact appearance.
                """
            
            response = self.model.generate_content([prompt, image])
            return response.text
            
        except Exception as e:
            raise RuntimeError(f"Image text extraction failed: {str(e)}")
    
    def extract_mathematical_content(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Extract and structure mathematical content from image.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Dictionary with structured mathematical content
        """
        if not self.available:
            raise RuntimeError(f"Gemini client not available: {self.error_message}")
        
        try:
            image = Image.open(BytesIO(image_bytes))
            
            prompt = """
            Analyze this mathematical image and extract all content in a structured format.
            
            Return a JSON response with the following structure:
            {
                "equations": ["list of equations found"],
                "expressions": ["list of mathematical expressions"],
                "steps": ["list of solution steps in order"],
                "variables": ["list of variables used"],
                "constants": ["list of constants/numbers"],
                "functions": ["list of functions used (sin, cos, etc.)"],
                "raw_text": "complete raw text extraction"
            }
            
            Ensure all mathematical notation is converted to plain text format that can be parsed by SymPy.
            """
            
            response = self.model.generate_content([prompt, image])
            
            # Try to parse JSON response
            import json
            try:
                return json.loads(response.text)
            except json.JSONDecodeError:
                # Fallback to simple text extraction
                return {
                    "equations": [],
                    "expressions": [],
                    "steps": [],
                    "variables": [],
                    "constants": [],
                    "functions": [],
                    "raw_text": response.text
                }
                
        except Exception as e:
            raise RuntimeError(f"Mathematical content extraction failed: {str(e)}")
    
    def analyze_math_solution(self, text: str, question_context: str = "") -> Dict[str, Any]:
        """
        Analyze a mathematical solution for correctness and quality.
        
        Args:
            text: The solution text to analyze
            question_context: Optional context about the question
            
        Returns:
            Dictionary containing analysis results
        """
        if not self.available:
            raise RuntimeError(f"Gemini client not available: {self.error_message}")
        
        prompt = f"""
        Analyze this mathematical solution for correctness, clarity, and completeness.
        
        Question Context: {question_context}
        
        Solution: {text}
        
        Please provide your analysis in JSON format with the following structure:
        {{
            "score": <number 0-10>,
            "feedback": "<detailed feedback>",
            "criteria": {{
                "correctness": {{"score": <0-10>, "comment": "<comment>"}},
                "methodology": {{"score": <0-10>, "comment": "<comment>"}},
                "clarity": {{"score": <0-10>, "comment": "<comment>"}},
                "completeness": {{"score": <0-10>, "comment": "<comment>"}}
            }},
            "suggestions": ["<suggestion1>", "<suggestion2>"]
        }}
        """
        
        try:
            response = self.model.generate_content(prompt)
            # Try to parse JSON response
            import json
            try:
                return json.loads(response.text)
            except json.JSONDecodeError:
                # If JSON parsing fails, return structured fallback
                return {
                    "score": 7,
                    "feedback": response.text,
                    "criteria": {},
                    "suggestions": []
                }
        except Exception as e:
            raise RuntimeError(f"Math solution analysis failed: {str(e)}")


@st.cache_resource
def get_gemini_client() -> Optional[GeminiClient]:
    """
    Get cached Gemini client instance.
    
    Returns:
        GeminiClient instance or None if not available
    """
    try:
        client = GeminiClient()
        if client.is_available():
            return client
    except Exception:
        pass
    return None
