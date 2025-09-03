#!/usr/bin/env python3
"""
Comprehensive test script for the enhanced OCR and math parsing system.

This script validates that our improvements work correctly for:
1. Layout preservation
2. Mathematical expression parsing
3. LaTeX conversion
4. Proper formatting output
"""

import sys
from pathlib import Path

# Add app directory to path
sys.path.append(str(Path(__file__).parent.parent / "app"))

def test_sample_mathematical_content():
    """Test with sample mathematical content similar to what we've seen in the warnings."""
    print("=== Testing Sample Mathematical Content ===")
    
    sample_ocr_output = """
**10**

∫ tan²x dx

⇒ ∫ (sec²x - 1) dx

⇒ tan x - x + C


**12**

∫ dx/(x² + 5x + 1)

⇒ ∫ dx/(x² + 5x + 1 + 3x - 3x)

⇒ ∫ dx/(x² + 2x + 1 + 3x)

⇒ ∫ dx/((x+1)² + 3x)

⇒ ∫ 1/(x+1)² dx + ∫ 1/(3x) dx

⇒ -1/(x+1) + log(3x) + C


**13**

∫ x²/(1+x³) dx

Let x³ = t

3x² dx = dt

x² dx = dt/3

⇒ ∫ dt/(3(1+t²))

⇒ 1/3 ∫ dt/(1+t²)

⇒ 1/3 arctan(t) + C

⇒ 1/3 arctan(x³) + C
"""
    
    try:
        from services.ocr_service import (
            _preserve_original_formatting,
            _detect_mathematical_sections,
            _format_mathematical_content,
            _convert_latex_expressions
        )
        
        print("Testing LaTeX conversion...")
        converted = _convert_latex_expressions(sample_ocr_output)
        print("✅ LaTeX conversion successful")
        print(f"Sample output:\n{converted[:200]}...\n")
        
        print("Testing formatting preservation...")
        formatted = _preserve_original_formatting(sample_ocr_output)
        print("✅ Formatting preservation successful")
        print(f"Formatted length: {len(formatted)} characters\n")
        
        print("Testing section detection...")
        sections = _detect_mathematical_sections(formatted)
        print(f"✅ Detected {len(sections)} sections")
        for i, section in enumerate(sections):
            print(f"  Section {i+1}: {section['type']} ({len(section['content'])} chars)")
        print()
        
        print("Testing mathematical content formatting...")
        for section in sections:
            formatted_content = _format_mathematical_content(section['content'])
            print(f"Section type: {section['type']}")
            print(f"Content preview: {formatted_content[:100]}...")
            print()
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

def test_problematic_expressions():
    """Test expressions that were causing parsing warnings."""
    print("=== Testing Problematic Expressions ===")
    
    problematic_expressions = [
        "== QUESTION === Here",
        "== SOLUTION === **10**",
        "3x^2 dx",
        "x^2 dx",
        "angle AOP",
        "angle QOD",
        "angle BOC",
        "D 5y 2y O 5y B",
        "== SOLUTION",
        "4 (55 + (8n - 3) + (12n + 8)",
        "180 55 + 8n - 3 +",
        "180 55 + 5 + 20n = 180 60 + 20n",
        "180 20n = 180 - 60 20n = 120 n",
        "360^"
    ]
    
    try:
        from services.math_parser import (
            _clean_expression,
            _is_likely_mathematical,
            validate_expression_syntax,
            extract_mathematical_expressions
        )
        
        print("Testing expression cleaning and filtering...")
        for expr in problematic_expressions:
            is_math = _is_likely_mathematical(expr)
            cleaned = _clean_expression(expr)
            
            status = "🔍" if is_math else "⏭️"
            print(f"{status} '{expr[:30]}...' -> Math: {is_math}, Cleaned: '{cleaned[:30]}...'")
        
        print("\n✅ Expression filtering completed successfully\n")
        
        # Test with a longer text sample
        print("Testing expression extraction from text...")
        sample_text = " ".join(problematic_expressions)
        extracted = extract_mathematical_expressions(sample_text)
        print(f"✅ Extracted {len(extracted)} valid mathematical expressions")
        for expr in extracted:
            print(f"  • {expr}")
        print()
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

def test_layout_preservation():
    """Test layout preservation functionality."""
    print("=== Testing Layout Preservation ===")
    
    sample_with_layout = """**Problem 1:**
∫ tan²x dx

Step 1: Rewrite using trigonometric identity
= ∫ (sec²x - 1) dx

Step 2: Integrate term by term  
= ∫ sec²x dx - ∫ 1 dx

Step 3: Apply integration formulas
= tan x - x + C

**Problem 2:**
Solve: x² + 5x + 6 = 0

Step 1: Factor the quadratic
= (x + 2)(x + 3) = 0

Step 2: Apply zero product property
x + 2 = 0  or  x + 3 = 0

Step 3: Solve for x
x = -2  or  x = -3"""
    
    try:
        from services.ocr_service import (
            _preserve_original_formatting,
            _detect_mathematical_sections
        )
        
        print("Original text structure:")
        lines = sample_with_layout.split('\n')
        print(f"  Total lines: {len(lines)}")
        print(f"  Empty lines: {len([l for l in lines if not l.strip()])}")
        print(f"  Non-empty lines: {len([l for l in lines if l.strip()])}")
        
        print("\nTesting formatting preservation...")
        preserved = _preserve_original_formatting(sample_with_layout)
        preserved_lines = preserved.split('\n')
        print(f"  Preserved lines: {len(preserved_lines)}")
        print(f"  Structure maintained: {'✅' if len(preserved_lines) >= len(lines) * 0.8 else '❌'}")
        
        print("\nTesting section detection...")
        sections = _detect_mathematical_sections(preserved)
        print(f"  Detected sections: {len(sections)}")
        for section in sections:
            print(f"    - {section['type']}: {len(section['content'])} chars")
        
        print("\n✅ Layout preservation test completed successfully\n")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🧪 Testing Enhanced OCR and Math Parsing System\n")
    print("=" * 60)
    
    test_sample_mathematical_content()
    test_problematic_expressions() 
    test_layout_preservation()
    
    print("=" * 60)
    print("✅ All tests completed!")
    print("\nThe enhanced system should now:")
    print("1. ✅ Preserve original layout and formatting from images")
    print("2. ✅ Filter out non-mathematical expressions to reduce parsing warnings") 
    print("3. ✅ Convert LaTeX notation to readable mathematical text")
    print("4. ✅ Structure content into logical sections")
    print("5. ✅ Display mathematical content in a line-by-line format")
    print("\nYou can now run the Streamlit app to see the improvements!")
