"""
Geometry problem parser and solver for angle calculations.

Provides functionality to parse geometry problems involving angles
and solve for unknown variables using angle sum rules.
"""

import sympy as sp
from sympy import symbols, Eq, solve, simplify
from typing import Dict, List, Any, Optional, Tuple
import re
from .math_parser import parse_to_sympy


def solve_angle_problem(angle_expressions: Dict[str, str], 
                       constraints: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Solve geometry problem with angle expressions and constraints.
    
    Args:
        angle_expressions: Dictionary mapping angle names to expressions
                          e.g., {"AOP": "5*y", "QOD": "2*y", "BOC": "5*y"}
        constraints: List of constraint strings (e.g., "straight_line", "full_circle")
        
    Returns:
        Dictionary containing solution and analysis
    """
    try:
        # Parse angle expressions
        parsed_angles = {}
        variables = set()
        
        for angle_name, expr_str in angle_expressions.items():
            parsed = parse_to_sympy(expr_str)
            if parsed is not None:
                parsed_angles[angle_name] = parsed
                variables.update(parsed.free_symbols)
            else:
                return {
                    "success": False,
                    "error": f"Could not parse angle expression for {angle_name}: {expr_str}"
                }
        
        if not parsed_angles:
            return {"success": False, "error": "No valid angle expressions found"}
        
        # Generate constraint equations
        equations = []
        
        # Apply automatic constraints based on angle relationships
        auto_constraints = _detect_angle_constraints(list(angle_expressions.keys()))
        equations.extend(auto_constraints)
        
        # Apply explicit constraints if provided
        if constraints:
            for constraint in constraints:
                constraint_eq = _parse_constraint(constraint, parsed_angles)
                if constraint_eq:
                    equations.append(constraint_eq)
        
        # Solve the system of equations
        if equations and variables:
            var_list = list(variables)
            solutions = solve(equations, var_list)
            
            # Calculate angle values
            angle_values = {}
            if solutions:
                for angle_name, expr in parsed_angles.items():
                    if isinstance(solutions, dict):
                        value = expr.subs(solutions)
                    else:
                        # Multiple solutions case
                        value = expr.subs(dict(zip(var_list, solutions[0])) if solutions else {})
                    
                    angle_values[angle_name] = float(value) if value.is_number else value
            
            return {
                "success": True,
                "solutions": solutions,
                "angle_values": angle_values,
                "equations": [str(eq) for eq in equations],
                "variables": [str(var) for var in variables]
            }
        else:
            return {
                "success": False,
                "error": "No constraints or variables found to solve"
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": f"Error solving angle problem: {str(e)}"
        }


def _detect_angle_constraints(angle_names: List[str]) -> List[sp.Eq]:
    """
    Automatically detect angle constraints based on naming patterns.
    
    Args:
        angle_names: List of angle names
        
    Returns:
        List of SymPy equations representing constraints
    """
    equations = []
    
    # Example: If we have angles around a point, they sum to 360°
    # This is a simple heuristic - in practice, you'd need more sophisticated pattern recognition
    
    # Look for patterns indicating angles around a point
    if _angles_around_point(angle_names):
        # Create equation: sum of angles = 360
        # This would need the actual parsed expressions, so we'll create a placeholder
        pass
    
    # Look for patterns indicating angles on a straight line
    if _angles_on_line(angle_names):
        # Create equation: sum of angles = 180
        pass
    
    return equations


def _angles_around_point(angle_names: List[str]) -> bool:
    """Check if angles appear to be around a single point."""
    # Simple heuristic: if all angles share a common vertex
    # Look for patterns like "AOP", "POB", "BOC" etc.
    if len(angle_names) < 3:
        return False
    
    # Extract potential center points (middle letters in 3-letter angle names)
    centers = []
    for name in angle_names:
        if len(name) == 3:
            centers.append(name[1])
    
    # If all angles share the same center, they might be around a point
    return len(set(centers)) == 1 and len(centers) >= 3


def _angles_on_line(angle_names: List[str]) -> bool:
    """Check if angles appear to be on a straight line."""
    # Simple heuristic: look for adjacent angles
    # This would need more sophisticated analysis in practice
    return len(angle_names) == 2


def _parse_constraint(constraint: str, parsed_angles: Dict[str, sp.Expr]) -> Optional[sp.Eq]:
    """
    Parse a constraint string into a SymPy equation.
    
    Args:
        constraint: Constraint description
        parsed_angles: Dictionary of parsed angle expressions
        
    Returns:
        SymPy equation or None if parsing fails
    """
    constraint = constraint.lower().strip()
    
    if constraint == "straight_line" or constraint == "linear":
        # Angles on a straight line sum to 180°
        total = sum(parsed_angles.values())
        return Eq(total, 180)
    
    elif constraint == "full_circle" or constraint == "around_point":
        # Angles around a point sum to 360°
        total = sum(parsed_angles.values())
        return Eq(total, 360)
    
    elif constraint.startswith("sum="):
        # Custom sum constraint
        try:
            target_sum = float(constraint.split("=")[1])
            total = sum(parsed_angles.values())
            return Eq(total, target_sum)
        except:
            return None
    
    elif "=" in constraint:
        # Custom equation
        try:
            left, right = constraint.split("=", 1)
            left_expr = parse_to_sympy(left.strip())
            right_expr = parse_to_sympy(right.strip())
            if left_expr is not None and right_expr is not None:
                return Eq(left_expr, right_expr)
        except:
            return None
    
    return None


def analyze_triangle_angles(angles: Dict[str, str]) -> Dict[str, Any]:
    """
    Analyze triangle angle relationships.
    
    Args:
        angles: Dictionary of angle expressions for a triangle
        
    Returns:
        Analysis results including angle sum check
    """
    if len(angles) != 3:
        return {
            "success": False,
            "error": "Triangle must have exactly 3 angles"
        }
    
    # Triangle angles sum to 180°
    result = solve_angle_problem(angles, ["sum=180"])
    
    if result["success"]:
        angle_values = result.get("angle_values", {})
        
        # Additional triangle analysis
        analysis = {
            "triangle_type": _classify_triangle(angle_values),
            "angle_sum": sum(angle_values.values()) if all(isinstance(v, (int, float)) for v in angle_values.values()) else "unknown",
            "largest_angle": max(angle_values.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0),
            "smallest_angle": min(angle_values.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0)
        }
        
        result.update(analysis)
    
    return result


def _classify_triangle(angles: Dict[str, Any]) -> str:
    """Classify triangle based on angle measures."""
    if not all(isinstance(v, (int, float)) for v in angles.values()):
        return "unknown"
    
    angle_values = list(angles.values())
    angle_values.sort()
    
    # Check for right triangle
    if any(abs(angle - 90) < 0.1 for angle in angle_values):
        return "right"
    
    # Check for obtuse triangle
    if any(angle > 90 for angle in angle_values):
        return "obtuse"
    
    # Check for acute triangle
    if all(angle < 90 for angle in angle_values):
        return "acute"
    
    return "unknown"


def solve_sample_problem() -> Dict[str, Any]:
    """
    Solve a sample geometry problem for testing.
    
    Example problem: 
    In a figure, angles AOP, QOD, and BOC meet at point O.
    If AOP = 5y, QOD = 2y, and BOC = 5y, and they form a straight line,
    find the value of y and all angles.
    
    Returns:
        Solution dictionary
    """
    # Define the angle expressions
    angles = {
        "AOP": "5*y",
        "QOD": "2*y", 
        "BOC": "5*y"
    }
    
    # These angles are on a straight line, so they sum to 180°
    constraints = ["sum=180"]
    
    result = solve_angle_problem(angles, constraints)
    
    # Add interpretation
    if result["success"]:
        y_value = result["solutions"].get(symbols('y'), "unknown")
        result["interpretation"] = f"y = {y_value}, so the angles are: "
        
        angle_values = result.get("angle_values", {})
        for name, value in angle_values.items():
            result["interpretation"] += f"{name} = {value}°, "
        
        result["interpretation"] = result["interpretation"].rstrip(", ")
    
    return result
