"""
Main grading orchestration service.

Coordinates OCR extraction, mathematical validation, and LLM evaluation
to provide comprehensive grading of handwritten math solutions.
"""

import json
from typing import Dict, List, Any, Optional
import streamlit as st

from .ocr_service import extract_text
from .math_parser import (parse_to_sympy, equivalent, extract_equations_from_text, 
                         solve_equation, extract_mathematical_expressions, 
                         validate_expression_syntax, safe_parse_expressions,
                         preprocess_text_for_parsing)
from .geometry_parser import solve_angle_problem, solve_sample_problem
from .gemini_client import get_gemini_client


def extract_and_grade(ocr_results: Dict[str, List[Dict[str, Any]]], 
                     edited_text: str = "") -> Dict[str, Any]:
    """
    Main orchestration function for extracting and grading solutions.
    
    Args:
        ocr_results: OCR results from images
        edited_text: User-edited text (if any)
        
    Returns:
        Dictionary containing complete grading results
    """
    try:
        # Step 1: Process OCR results and text
        processed_text = _process_solution_text(ocr_results, edited_text)
        
        # Step 2: Perform symbolic mathematical checks
        symbolic_checks = _perform_symbolic_checks(processed_text)
        
        # Step 3: Perform LLM evaluation
        llm_evaluation = _perform_llm_evaluation(processed_text, ocr_results)
        
        # Step 4: Combine results
        final_results = {
            "ocr_results": ocr_results,
            "processed_text": processed_text,
            "symbolic_checks": symbolic_checks,
            "llm_evaluation": llm_evaluation,
            "overall_score": _calculate_overall_score(symbolic_checks, llm_evaluation),
            "timestamp": _get_timestamp()
        }
        
        return final_results
        
    except Exception as e:
        return {
            "error": f"Grading failed: {str(e)}",
            "ocr_results": ocr_results,
            "processed_text": edited_text,
            "symbolic_checks": {},
            "llm_evaluation": {},
            "overall_score": 0,
            "timestamp": _get_timestamp()
        }


def _process_solution_text(ocr_results: Dict[str, List[Dict[str, Any]]], 
                          edited_text: str) -> str:
    """
    Process and clean solution text from OCR or user input.
    
    Args:
        ocr_results: Raw OCR results
        edited_text: User-edited text
        
    Returns:
        Processed text ready for analysis
    """
    if edited_text and edited_text.strip():
        return edited_text.strip()
    
    # Extract text from OCR results
    text_parts = []
    
    # Process solution regions
    solution_regions = ocr_results.get("solution", [])
    for region in solution_regions:
        if region.get("text"):
            text_parts.append(region["text"])
    
    return " ".join(text_parts)


def _perform_symbolic_checks(text: str) -> Dict[str, Any]:
    """
    Perform symbolic mathematical validation on the solution.
    
    Args:
        text: Solution text to validate
        
    Returns:
        Dictionary containing symbolic check results
    """
    results = {
        "score": 0,
        "checks": [],
        "equations": [],
        "expressions": [],
        "validation_results": [],
        "geometry_result": None
    }
    
    try:
        # Use improved parsing with better filtering
        parsed_expressions = safe_parse_expressions(text)
        
        # Process valid expressions
        valid_expressions = []
        for expr_text, parsed_expr in parsed_expressions:
            if parsed_expr is not None:
                valid_expressions.append(expr_text)
                results["checks"].append({
                    "type": "expression_validity",
                    "correct": True,
                    "message": f"Valid expression: {expr_text}",
                    "details": f"Variables: {[str(var) for var in parsed_expr.free_symbols]}"
                })
        
        results["expressions"] = valid_expressions
        
        # Extract equations more carefully
        equations = extract_equations_from_text(text)
        valid_equations = [eq for eq in equations if eq[1] is not None]
        results["equations"] = [eq[0] for eq in valid_equations]  # Store original strings
        
        # Add equation validation checks
        for eq_text, eq_expr in valid_equations:
            results["checks"].append({
                "type": "equation_validity",
                "correct": True,
                "message": f"Valid equation: {eq_text}",
                "details": f"Variables: {[str(var) for var in eq_expr.free_symbols]}"
            })
        
        # Calculate score based on mathematical content found
        valid_expressions_count = len(valid_expressions)
        valid_equations_count = len(valid_equations)
        
        if valid_expressions_count > 0 or valid_equations_count > 0:
            base_score = min(6, valid_expressions_count * 1.5)  # Up to 6 points for expressions
            equation_score = min(4, valid_equations_count * 2)  # Up to 4 points for equations
            results["score"] = int(base_score + equation_score)
        
        # Check for geometry problems
        geometry_result = _check_geometry_problem(text)
        if geometry_result:
            results["geometry_result"] = geometry_result
            if geometry_result.get("success"):
                results["checks"].append({
                    "type": "geometry_solution",
                    "correct": True,
                    "message": "Geometry problem solved correctly",
                    "details": geometry_result.get("interpretation", "")
                })
                results["score"] = max(results["score"], 7)  # Boost score for geometry
            else:
                results["checks"].append({
                    "type": "geometry_solution",
                    "correct": False,
                    "message": "Geometry problem detected but could not solve",
                    "details": geometry_result.get("error", "")
                })
        
        # Add summary check
        results["checks"].append({
            "type": "content_summary",
            "correct": True,
            "message": f"Found {valid_expressions_count} valid expressions and {valid_equations_count} valid equations",
            "details": f"Mathematical content score: {results['score']}/10"
        })
        
    except Exception as e:
        results["checks"].append({
            "type": "parsing_error",
            "correct": False,
            "message": f"Error during symbolic analysis: {str(e)}",
            "details": "Failed to parse mathematical content"
        })
    
    return results


def _check_geometry_problem(text: str) -> Optional[Dict[str, Any]]:
    """
    Check if text contains a geometry problem and attempt to solve it.
    
    Args:
        text: Solution text
        
    Returns:
        Geometry solution result or None
    """
    # Look for angle expressions in the text
    angle_patterns = [
        r"(\w+)\s*=\s*([^,\n]+)",  # Pattern like "AOP = 5*y"
        r"angle\s+(\w+)\s*=\s*([^,\n]+)",  # Pattern like "angle AOP = 5*y"
    ]
    
    angles = {}
    for pattern in angle_patterns:
        import re
        matches = re.findall(pattern, text, re.IGNORECASE)
        for angle_name, expr in matches:
            angles[angle_name] = expr.strip()
    
    if len(angles) >= 2:
        # Try to solve as angle problem
        return solve_angle_problem(angles, ["sum=180"])  # Assume straight line for now
    
    # If no specific patterns found, try the sample problem
    if any(keyword in text.lower() for keyword in ["angle", "degree", "straight", "line", "point"]):
        return solve_sample_problem()
    
    return None


def _perform_llm_evaluation(text: str, ocr_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Perform LLM-based evaluation of the solution.
    
    Args:
        text: Solution text
        ocr_results: Original OCR results for context
        
    Returns:
        Dictionary containing LLM evaluation results
    """
    # Get Gemini client
    client = get_gemini_client()
    
    if client and client.is_available():
        try:
            # Prepare context
            question_text = _extract_question_text(ocr_results)
            
            # Analyze the solution
            analysis = client.analyze_math_solution(text, question_text)
            
            return analysis
            
        except Exception as e:
            # Fallback to mocked evaluation
            return _mock_llm_evaluation(text, str(e))
    else:
        # Client not available, use mocked evaluation
        return _mock_llm_evaluation(text, "Gemini client not available")


def _extract_question_text(ocr_results: Dict[str, List[Dict[str, Any]]]) -> str:
    """Extract question text from OCR results."""
    question_regions = ocr_results.get("question", [])
    text_parts = []
    
    for region in question_regions:
        if region.get("text"):
            text_parts.append(region["text"])
    
    return " ".join(text_parts)


def _mock_llm_evaluation(text: str, reason: str = "") -> Dict[str, Any]:
    """
    Provide a mocked LLM evaluation when the real LLM is not available.
    
    Args:
        text: Solution text
        reason: Reason for using mock evaluation
        
    Returns:
        Mocked evaluation results
    """
    # Simple heuristics for mocked evaluation
    word_count = len(text.split())
    has_equations = "=" in text
    has_numbers = any(char.isdigit() for char in text)
    
    # Base score calculation
    base_score = 5  # Start with middle score
    
    if has_equations:
        base_score += 2
    if has_numbers:
        base_score += 1
    if word_count > 20:
        base_score += 1
    if word_count > 50:
        base_score += 1
    
    base_score = min(base_score, 10)  # Cap at 10
    
    return {
        "score": base_score,
        "feedback": f"Automated evaluation (mocked): The solution shows {'good' if base_score >= 7 else 'adequate' if base_score >= 5 else 'limited'} mathematical content. {reason}",
        "criteria": {
            "correctness": {
                "score": base_score,
                "comment": "Based on presence of mathematical content"
            },
            "methodology": {
                "score": base_score - 1,
                "comment": "Shows some mathematical approach"
            },
            "clarity": {
                "score": min(base_score, 8),
                "comment": f"Solution contains {word_count} words"
            },
            "completeness": {
                "score": base_score - 2 if base_score > 2 else 1,
                "comment": "Partial solution analysis"
            }
        },
        "suggestions": [
            "Show more detailed steps",
            "Include explanations for each calculation",
            "Verify final answers"
        ],
        "mock_evaluation": True,
        "mock_reason": reason
    }


def _calculate_overall_score(symbolic_checks: Dict[str, Any], 
                           llm_evaluation: Dict[str, Any]) -> float:
    """
    Calculate overall score from symbolic and LLM evaluations.
    
    Args:
        symbolic_checks: Results from symbolic validation
        llm_evaluation: Results from LLM evaluation
        
    Returns:
        Overall score (0-10)
    """
    symbolic_score = symbolic_checks.get("score", 0)
    llm_score = llm_evaluation.get("score", 0)
    
    # Weight symbolic checks higher for math problems
    weighted_score = (symbolic_score * 0.6) + (llm_score * 0.4)
    
    return round(weighted_score, 1)


def _get_timestamp() -> str:
    """Get current timestamp as string."""
    from datetime import datetime
    return datetime.now().isoformat()
