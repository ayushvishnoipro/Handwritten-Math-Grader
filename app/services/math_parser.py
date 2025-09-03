"""
Mathematical expression parser using SymPy for symbolic validation.

Provides functions to parse mathematical expressions from text
and verify their symbolic equivalence.
"""

import sympy as sp
from sympy import symbols, sympify, simplify, N
from sympy.parsing.latex import parse_latex
from typing import Any, Dict, List, Optional, Union, Tuple
import re
import logging


def parse_to_sympy(expr_str: str) -> Optional[sp.Expr]:
    """
    Parse a mathematical expression string to SymPy expression.
    
    Args:
        expr_str: String representation of mathematical expression
        
    Returns:
        SymPy expression or None if parsing fails
    """
    if not expr_str or not expr_str.strip():
        return None
    
    # Early filtering for obviously non-mathematical content
    if not _is_likely_mathematical(expr_str):
        return None
    
    # Clean and normalize the expression
    cleaned_expr = _clean_expression(expr_str)
    
    # Skip if cleaning resulted in empty string
    if not cleaned_expr or not cleaned_expr.strip():
        return None
    
    # Skip expressions that contain differential notation (dx, dt, etc.)
    if re.search(r'd[a-zA-Z](?!\w)', cleaned_expr):
        return None
    
    # Skip expressions that are too complex or contain problematic patterns
    skip_patterns = ['integral', '=>', 'arctan', 'sec', 'angle', 'EOF', 'unexpected']
    if any(pattern in cleaned_expr.lower() for pattern in skip_patterns):
        return None
    
    # Skip very short expressions that are likely fragments
    if len(cleaned_expr.strip()) < 3:
        return None
    
    # Skip expressions with unbalanced parentheses
    if cleaned_expr.count('(') != cleaned_expr.count(')'):
        return None
    
    # Skip incomplete expressions (ending with operators)
    if re.search(r'[\+\-\*\/\^]$', cleaned_expr.strip()):
        return None
    
    try:
        # Try parsing as regular expression first
        expr = sympify(cleaned_expr, evaluate=False)
        return expr
    except Exception as e:
        # Log the specific error for debugging but don't spam warnings
        if "invalid syntax" not in str(e).lower():
            logging.debug(f"Failed to parse expression '{expr_str}': {e}")
    
    # Try with common mathematical substitutions
    try:
        substituted = _apply_common_substitutions(cleaned_expr)
        if substituted != cleaned_expr:  # Only if substitution changed something
            expr = sympify(substituted, evaluate=False)
            return expr
    except Exception:
        pass
    
    return None


def equivalent(expr1: Union[str, sp.Expr], expr2: Union[str, sp.Expr], 
               tolerance: float = 1e-10) -> bool:
    """
    Check if two mathematical expressions are equivalent.
    
    Args:
        expr1: First expression (string or SymPy expression)
        expr2: Second expression (string or SymPy expression)
        tolerance: Numerical tolerance for comparison
        
    Returns:
        True if expressions are equivalent, False otherwise
    """
    # Convert to SymPy expressions
    if isinstance(expr1, str):
        expr1 = parse_to_sympy(expr1)
    if isinstance(expr2, str):
        expr2 = parse_to_sympy(expr2)
    
    if expr1 is None or expr2 is None:
        return False
    
    try:
        # Try symbolic equivalence first
        difference = simplify(expr1 - expr2)
        if difference == 0:
            return True
        
        # Try numerical equivalence with different variable values
        if _numerical_equivalence_check(expr1, expr2, tolerance):
            return True
        
        # Try expanding and simplifying both expressions
        expanded1 = sp.expand(expr1)
        expanded2 = sp.expand(expr2)
        if simplify(expanded1 - expanded2) == 0:
            return True
            
        return False
        
    except Exception as e:
        logging.warning(f"Error checking equivalence: {e}")
        return False


def extract_equations_from_text(text: str) -> List[Tuple[str, sp.Expr]]:
    """
    Extract mathematical equations from text.
    
    Args:
        text: Input text containing equations
        
    Returns:
        List of tuples (original_string, sympy_expression)
    """
    equations = []
    
    # Pattern for simple equations (contains = sign but not complex text)
    equation_patterns = [
        r'([a-zA-Z0-9\s\+\-\*\/\^\(\)\=]{5,30}=[a-zA-Z0-9\s\+\-\*\/\^\(\)]{1,20})',  # Simple equations
        r'([xy]\s*=\s*\d+)',  # Variable assignments like x = 5
        r'(\d+[xy]\s*=\s*\d+)',  # Coefficient equations like 5x = 10
    ]
    
    for pattern in equation_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            cleaned_match = match.strip()
            if cleaned_match and '=' in cleaned_match:
                # Split equation and try to parse both sides
                try:
                    left, right = cleaned_match.split('=', 1)
                    left_expr = parse_to_sympy(left.strip())
                    right_expr = parse_to_sympy(right.strip())
                    
                    if left_expr is not None and right_expr is not None:
                        # Create equation as left - right = 0
                        equation = left_expr - right_expr
                        equations.append((cleaned_match, equation))
                except:
                    continue
    
    return equations


def solve_equation(equation: Union[str, sp.Expr], 
                  variables: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Solve a mathematical equation.
    
    Args:
        equation: Equation to solve (string or SymPy expression)
        variables: List of variable names to solve for
        
    Returns:
        Dictionary containing solution information
    """
    if isinstance(equation, str):
        equation = parse_to_sympy(equation)
    
    if equation is None:
        return {"success": False, "error": "Could not parse equation"}
    
    try:
        # Extract variables if not provided
        if variables is None:
            variables = [str(var) for var in equation.free_symbols]
        
        # Convert variable names to SymPy symbols
        var_symbols = [symbols(var) for var in variables]
        
        # Solve the equation
        if len(var_symbols) == 1:
            # Single variable equation
            solutions = sp.solve(equation, var_symbols[0])
        else:
            # Multiple variables
            solutions = sp.solve(equation, var_symbols)
        
        return {
            "success": True,
            "solutions": solutions,
            "variables": variables,
            "equation": str(equation)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "equation": str(equation)
        }


def evaluate_at_values(expr: Union[str, sp.Expr], 
                      values: Dict[str, float]) -> Optional[float]:
    """
    Evaluate expression at specific variable values.
    
    Args:
        expr: Expression to evaluate
        values: Dictionary mapping variable names to values
        
    Returns:
        Numerical result or None if evaluation fails
    """
    if isinstance(expr, str):
        expr = parse_to_sympy(expr)
    
    if expr is None:
        return None
    
    try:
        # Substitute values
        substituted = expr.subs(values)
        
        # Convert to numerical value
        result = N(substituted)
        
        if result.is_real:
            return float(result)
        else:
            return complex(result)
            
    except Exception as e:
        logging.warning(f"Error evaluating expression: {e}")
        return None


def extract_mathematical_expressions(text: str) -> List[str]:
    """
    Extract mathematical expressions from text with improved pattern matching.
    
    Args:
        text: Input text that may contain mathematical expressions
        
    Returns:
        List of mathematical expression strings
    """
    expressions = []
    
    # Pattern for mathematical expressions
    math_patterns = [
        # Equations with equals sign
        r'[a-zA-Z0-9\+\-\*\/\^\(\)\s]*\s*=\s*[a-zA-Z0-9\+\-\*\/\^\(\)\s]+',
        # Expressions with variables and operations
        r'[a-zA-Z][a-zA-Z0-9]*\s*[\+\-\*\/\^]\s*[a-zA-Z0-9\+\-\*\/\^\(\)\s]+',
        # Trigonometric expressions
        r'(sin|cos|tan|sec|csc|cot|arcsin|arccos|arctan)\s*\([^)]+\)',
        # Logarithmic expressions
        r'(log|ln)\s*\([^)]+\)',
        # Square root expressions
        r'sqrt\s*\([^)]+\)',
        # Expressions with integrals
        r'integral\s+of\s+[^d]+\s+d[a-zA-Z]',
        # Fractions
        r'\([^)]+\)\s*/\s*\([^)]+\)',
        # Powers and exponents
        r'[a-zA-Z0-9]+\s*\*\*\s*[a-zA-Z0-9\(\)]+',
        # Simple algebraic expressions
        r'[a-zA-Z0-9]+\s*[\+\-\*\/]\s*[a-zA-Z0-9]+',
    ]
    
    for pattern in math_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            cleaned = _clean_expression(match)
            if cleaned and len(cleaned) > 2:  # Must be substantial
                expressions.append(cleaned)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_expressions = []
    for expr in expressions:
        if expr not in seen:
            seen.add(expr)
            unique_expressions.append(expr)
    
    return unique_expressions


def validate_expression_syntax(expr_str: str) -> Dict[str, Any]:
    """
    Validate the syntax of a mathematical expression.
    
    Args:
        expr_str: Mathematical expression string
        
    Returns:
        Dictionary with validation results
    """
    result = {
        "is_valid": False,
        "errors": [],
        "warnings": [],
        "cleaned_expression": "",
        "variables": [],
        "expression_type": "unknown"
    }
    
    if not expr_str or not expr_str.strip():
        result["errors"].append("Empty expression")
        return result
    
    try:
        # Clean the expression
        cleaned = _clean_expression(expr_str)
        result["cleaned_expression"] = cleaned
        
        if not cleaned:
            result["errors"].append("Expression cleaned to empty string")
            return result
        
        # Try to parse with SymPy
        parsed_expr = sympify(cleaned, evaluate=False)
        
        if parsed_expr is not None:
            result["is_valid"] = True
            result["variables"] = [str(var) for var in parsed_expr.free_symbols]
            
            # Determine expression type
            if "=" in expr_str:
                result["expression_type"] = "equation"
            elif any(func in cleaned.lower() for func in ["sin", "cos", "tan", "log", "ln", "sqrt"]):
                result["expression_type"] = "function"
            elif result["variables"]:
                result["expression_type"] = "algebraic"
            else:
                result["expression_type"] = "numeric"
        
    except Exception as e:
        result["errors"].append(f"Parsing error: {str(e)}")
    
    # Add warnings for potentially problematic expressions
    if len(cleaned) > 100:
        result["warnings"].append("Expression is very long")
    
    if cleaned.count("(") != cleaned.count(")"):
        result["warnings"].append("Unbalanced parentheses")
    
    return result


def preprocess_text_for_parsing(text: str) -> List[str]:
    """
    Preprocess text to extract potentially mathematical expressions.
    
    Args:
        text: Raw text that may contain mathematical content
        
    Returns:
        List of strings that might be mathematical expressions
    """
    if not text:
        return []
    
    # Split text into lines and clean each line
    lines = text.split('\n')
    potential_expressions = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Skip section headers
        if line.startswith('==') or line.endswith('==') or '===' in line:
            continue
        
        # Skip pure text descriptions
        if len(line) > 50 and not re.search(r'[=+\-*/^()]', line):
            continue
        
        # Split line by common delimiters that separate expressions
        # Don't split on mathematical operators, only on text separators
        parts = re.split(r'[.;,]\s+(?=[A-Z])', line)  # Split on punctuation followed by capital letter
        
        for part in parts:
            part = part.strip()
            if len(part) < 3:  # Skip very short parts
                continue
            
            # Check if this part looks mathematical
            if _is_likely_mathematical(part):
                potential_expressions.append(part)
    
    return potential_expressions


def safe_parse_expressions(text: str) -> List[Tuple[str, Optional[sp.Expr]]]:
    """
    Safely parse mathematical expressions from text with improved filtering.
    
    Args:
        text: Text containing mathematical expressions
        
    Returns:
        List of tuples (original_text, parsed_expression)
    """
    results = []
    
    # Preprocess text to get potential expressions
    potential_expressions = preprocess_text_for_parsing(text)
    
    for expr_text in potential_expressions:
        try:
            parsed = parse_to_sympy(expr_text)
            if parsed is not None:
                results.append((expr_text, parsed))
        except Exception as e:
            # Log at debug level to avoid spam
            logging.debug(f"Failed to parse '{expr_text}': {e}")
            continue
    
    return results


def _clean_expression(expr_str: str) -> str:
    """Clean and normalize expression string."""
    if not expr_str:
        return ""
    
    # Remove extra whitespace and normalize
    cleaned = re.sub(r'\s+', ' ', expr_str.strip())
    
    # Skip section headers and formatting early
    if cleaned.startswith('==') or cleaned.endswith('==') or '===' in cleaned:
        return ""
    
    # Skip angle notations and geometric terms
    if 'angle' in cleaned.lower():
        return ""
    
    # Skip incomplete expressions with dangling operators
    if re.search(r'[\+\-\*\/\^]\s*$', cleaned):
        return ""
    
    # Skip expressions that are clearly multi-line or contain line breaks
    if '\n' in cleaned or len(cleaned.split()) > 10:
        return ""
    
    # Convert LaTeX expressions to normal mathematical notation
    cleaned = _convert_latex_to_normal(cleaned)
    
    # Remove problematic characters and patterns
    cleaned = re.sub(r'[∠∆]', '', cleaned)  # Remove angle and triangle symbols
    cleaned = re.sub(r'[{}]', '', cleaned)  # Remove remaining braces
    cleaned = re.sub(r'[°]', '', cleaned)  # Remove degree symbols
    cleaned = re.sub(r'[""''`]', '', cleaned)  # Remove various quote marks
    
    # Skip if it contains text that's not mathematical
    problematic_words = [
        'given', 'find', 'value', 'figure', 'let', 'question', 'solution', 
        'here', 'rightarrow', 'therefore', 'hence', 'thus', 'where', 
        'when', 'answer', 'angle', 'from', 'the', 'and', 'but', 'with',
        'eof', 'unexpected', 'syntax', 'error', 'invalid', 'failed'
    ]
    if any(word in cleaned.lower() for word in problematic_words):
        return ""
    
    # Skip if it's just punctuation or comparison operators
    if cleaned.strip() in ['==', '=', '===', '!=', '**', '.', ',', ':', ';', '+', '-', '*', '/', '^']:
        return ""
    
    # Skip section headers and formatting
    if cleaned.startswith('==') or '===' in cleaned or cleaned.startswith('**') or cleaned.endswith('**'):
        return ""
    
    # Skip if it's just a number or single character
    if re.match(r'^\d+\.?\d*$', cleaned.strip()) or len(cleaned.strip()) == 1:
        return ""
    
    # Skip expressions that end with incomplete operators or parentheses
    if re.search(r'[\+\-\*\/\^]\s*$', cleaned) or cleaned.endswith('('):
        return ""
    
    # Replace common Unicode math symbols
    replacements = {
        '×': '*',
        '÷': '/',
        '²': '**2',
        '³': '**3',
        '⁴': '**4',
        '⁵': '**5',
        '√': 'sqrt',
        'π': 'pi',
        '∞': 'oo',
        '±': '+/-',
        '≤': '<=',
        '≥': '>=',
        '≠': '!=',
        '≈': '~',
        '∫': 'integral',
    }
    
    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)
    
    # Handle implicit multiplication (e.g., "2x" -> "2*x")
    cleaned = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', cleaned)
    cleaned = re.sub(r'([a-zA-Z])(\d)', r'\1*\2', cleaned)
    cleaned = re.sub(r'([a-zA-Z])([a-zA-Z])', r'\1*\2', cleaned)  # Variables like xy -> x*y
    
    # Clean up multiple operators
    cleaned = re.sub(r'\*{2,}', '**', cleaned)  # Fix multiple asterisks
    cleaned = re.sub(r'\+{2,}', '+', cleaned)   # Fix multiple plus signs
    cleaned = re.sub(r'-{2,}', '-', cleaned)    # Fix multiple minus signs
    
    # Final validation - must contain at least one mathematical operator or variable
    if not re.search(r'[=+\-*/^()]', cleaned) and not re.search(r'[a-zA-Z]', cleaned):
        return ""
    
    return cleaned


def _convert_latex_to_normal(text: str) -> str:
    """Convert LaTeX mathematical expressions to normal notation."""
    # Remove dollar signs and LaTeX environment markers
    text = re.sub(r'\$([^$]+)\$', r'\1', text)
    text = re.sub(r'\\begin\{[^}]+\}', '', text)
    text = re.sub(r'\\end\{[^}]+\}', '', text)
    
    # Convert common LaTeX commands to normal notation
    latex_conversions = {
        r'\\int': 'integral',
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
        r'\\sqrt\[(\d+)\]\{([^}]+)\}': r'(\2)**(1/\1)',  # nth root
        r'\\Rightarrow': '=>',
        r'\\rightarrow': '->',
        r'\\leftarrow': '<-',
        r'\\leftrightarrow': '<->',
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
        r'\\xi': 'xi',
        r'\\zeta': 'zeta',
        r'\\eta': 'eta',
        r'\\rho': 'rho',
        r'\\nu': 'nu',
        r'\\kappa': 'kappa',
        r'\\epsilon': 'epsilon',
        r'\\varepsilon': 'epsilon',
        r'\\sum': 'sum',
        r'\\prod': 'product',
        r'\\lim': 'limit',
        r'\\infty': 'oo',
        r'\\cdot': '*',
        r'\\times': '*',
        r'\\div': '/',
        r'\\pm': '+/-',
        r'\\mp': '-/+',
        r'\\leq': '<=',
        r'\\geq': '>=',
        r'\\neq': '!=',
        r'\\approx': '~',
        r'\\equiv': '==',
        r'\\left': '',
        r'\\right': '',
        r'\\\\': '',
    }
    
    for pattern, replacement in latex_conversions.items():
        text = re.sub(pattern, replacement, text)
    
    # Handle fractions more carefully (for cases without \\frac)
    text = re.sub(r'frac\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)', text)
    
    # Handle exponents
    text = re.sub(r'\^(\d+)', r'**\1', text)
    text = re.sub(r'\^\{([^}]+)\}', r'**(\1)', text)
    text = re.sub(r'\_(\d+)', r'_\1', text)  # Subscripts (keep as is for now)
    text = re.sub(r'\_\{([^}]+)\}', r'_(\1)', text)
    
    # Clean up any remaining braces and backslashes
    text = re.sub(r'[{}]', '', text)
    text = re.sub(r'\\[a-zA-Z]+', '', text)  # Remove any remaining LaTeX commands
    
    return text


def _looks_like_latex(expr_str: str) -> bool:
    """Check if expression looks like LaTeX."""
    latex_indicators = [
        r'\\frac', r'\\sqrt', r'\\sum', r'\\int', r'\\alpha', r'\\beta',
        r'\\gamma', r'\\theta', r'\\pi', r'\\sin', r'\\cos', r'\\tan',
        r'\\log', r'\\ln', r'\\exp', r'^{', r'_{', r'\\left', r'\\right'
    ]
    
    return any(indicator in expr_str for indicator in latex_indicators)


def _apply_common_substitutions(expr_str: str) -> str:
    """Apply common mathematical substitutions."""
    substitutions = {
        r'\bsin\b': 'sin',
        r'\bcos\b': 'cos',
        r'\btan\b': 'tan',
        r'\blog\b': 'log',
        r'\bln\b': 'log',
        r'\bexp\b': 'exp',
        r'\bsqrt\b': 'sqrt',
        r'\babs\b': 'Abs',
    }
    
    result = expr_str
    for pattern, replacement in substitutions.items():
        result = re.sub(pattern, replacement, result)
    
    return result


def _numerical_equivalence_check(expr1: sp.Expr, expr2: sp.Expr, 
                               tolerance: float = 1e-10) -> bool:
    """Check numerical equivalence by testing with random values."""
    variables = list(expr1.free_symbols.union(expr2.free_symbols))
    
    if not variables:
        # No variables, just compare numerical values
        try:
            val1 = float(N(expr1))
            val2 = float(N(expr2))
            return abs(val1 - val2) < tolerance
        except:
            return False
    
    # Test with multiple random values
    import random
    test_values = [-10, -1, 0, 1, 10]  # Common test points
    test_values.extend([random.uniform(-100, 100) for _ in range(5)])
    
    for _ in range(len(test_values)):
        try:
            # Create substitution dictionary
            values = {}
            for i, var in enumerate(variables):
                if i < len(test_values):
                    values[var] = test_values[i]
                else:
                    values[var] = random.uniform(-10, 10)
            
            val1 = evaluate_at_values(expr1, values)
            val2 = evaluate_at_values(expr2, values)
            
            if val1 is None or val2 is None:
                continue
                
            if abs(val1 - val2) > tolerance:
                return False
                
        except:
            continue
    
    return True


def _is_likely_mathematical(expr_str: str) -> bool:
    """
    Check if a string is likely to contain mathematical content.
    
    Args:
        expr_str: String to check
        
    Returns:
        True if the string likely contains mathematical content
    """
    if not expr_str or not expr_str.strip():
        return False
    
    # Convert to lowercase for checking
    text = expr_str.lower().strip()
    
    # Skip section headers and formatting
    if text.startswith('==') or text.endswith('==') or '===' in text:
        return False
    
    # Skip pure text descriptions
    non_math_phrases = [
        'question', 'solution', 'answer', 'given', 'find', 'figure', 
        'here', 'where', 'when', 'angle', 'therefore', 'hence', 'thus',
        'let', 'suppose', 'assume', 'consider', 'note', 'observe',
        'we have', 'we get', 'we can', 'this gives', 'substituting',
        'from the', 'in the', 'of the', 'to the', 'for the'
    ]
    
    # If it's mostly text phrases, skip it
    if any(phrase in text for phrase in non_math_phrases) and len(text) > 20:
        return False
    
    # Skip if it's just letters without mathematical operators
    if re.match(r'^[a-zA-Z\s]+$', text) and 'x' not in text and 'y' not in text:
        return False
    
    # Skip if it contains error messages
    error_indicators = ['eof', 'unexpected', 'syntax error', 'invalid', 'failed']
    if any(indicator in text for indicator in error_indicators):
        return False
    
    # Check for mathematical indicators
    math_indicators = [
        '=', '+', '-', '*', '/', '^', '(', ')', 
        'sin', 'cos', 'tan', 'log', 'ln', 'sqrt',
        'integral', 'dx', 'dy', 'dt', 'du',
        r'\d+x', r'\d+y', r'x\d+', r'y\d+',  # Variables with coefficients
        r'\d+\*', r'\*\d+',  # Multiplication
        r'\d+/', r'/\d+',    # Division
        r'\^\d+', r'\d+\^'   # Exponents
    ]
    
    # Check if it contains mathematical content
    has_math = False
    for indicator in math_indicators:
        if isinstance(indicator, str):
            if indicator in text:
                has_math = True
                break
        else:  # regex pattern
            if re.search(indicator, text):
                has_math = True
                break
    
    # Additional checks for variables and numbers
    if not has_math:
        # Check for single variables (x, y, z, etc.) with numbers
        if re.search(r'[a-z]\s*[=+\-*/^]\s*\d+', text):
            has_math = True
        # Check for equations
        elif '=' in text and re.search(r'\d', text):
            has_math = True
        # Check for expressions with variables
        elif re.search(r'[a-z][+\-*/^]', text) or re.search(r'[+\-*/^][a-z]', text):
            has_math = True
    
    return has_math
