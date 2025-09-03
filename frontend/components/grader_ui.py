"""
Grader UI component for displaying grading results and feedback.

Provides visualization of symbolic checks, LLM evaluation,
and overall grading results.
"""

import streamlit as st
import json
from typing import Dict, Any


def render_grader_ui(grading_results: Dict[str, Any]) -> None:
    """
    Render grading results interface.
    
    Args:
        grading_results: Dictionary containing grading evaluation results
    """
    if not grading_results:
        st.info("No grading results to display. Please grade the solution first.")
        return
    
    # Overall score display
    st.markdown("### 🏆 Grading Results")
    
    # Extract key metrics
    symbolic_checks = grading_results.get("symbolic_checks", {})
    llm_evaluation = grading_results.get("llm_evaluation", {})
    
    # Score display
    col1, col2, col3 = st.columns(3)
    
    with col1:
        symbolic_score = symbolic_checks.get("score", 0)
        st.metric(
            "🧮 Symbolic Score",
            f"{symbolic_score:.1f}/10",
            help="Score based on mathematical correctness"
        )
    
    with col2:
        llm_score = llm_evaluation.get("score", 0)
        st.metric(
            "🤖 LLM Score",
            f"{llm_score:.1f}/10",
            help="Score based on AI evaluation"
        )
    
    with col3:
        overall_score = (symbolic_score + llm_score) / 2
        st.metric(
            "📊 Overall Score",
            f"{overall_score:.1f}/10",
            help="Combined symbolic and LLM score"
        )
    
    # Detailed results in tabs
    tab1, tab2, tab3 = st.tabs(["🧮 Symbolic Checks", "🤖 LLM Evaluation", "📋 Raw Data"])
    
    with tab1:
        _render_symbolic_checks(symbolic_checks)
    
    with tab2:
        _render_llm_evaluation(llm_evaluation)
    
    with tab3:
        _render_raw_results(grading_results)


def _render_symbolic_checks(symbolic_checks: Dict[str, Any]) -> None:
    """
    Render symbolic validation results.
    
    Args:
        symbolic_checks: Dictionary containing symbolic check results
    """
    if not symbolic_checks:
        st.warning("No symbolic checks performed.")
        return
    
    st.write("**Mathematical Validation Results:**")
    
    # Check results
    checks = symbolic_checks.get("checks", [])
    if checks:
        for i, check in enumerate(checks):
            check_type = check.get("type", "unknown")
            is_correct = check.get("correct", False)
            message = check.get("message", "No message")
            
            status_icon = "✅" if is_correct else "❌"
            st.write(f"{status_icon} **{check_type}**: {message}")
            
            # Show details if available
            if check.get("details"):
                with st.expander(f"Details for {check_type}"):
                    st.code(check["details"], language=None)
    
    # Show equations if available
    equations = symbolic_checks.get("equations", [])
    if equations:
        st.write("**Detected Equations:**")
        for eq in equations:
            st.latex(eq)
    
    # Geometry results
    geometry_result = symbolic_checks.get("geometry_result")
    if geometry_result:
        st.write("**Geometry Solution:**")
        if geometry_result.get("solution"):
            for var, value in geometry_result["solution"].items():
                st.write(f"• {var} = {value}")


def _render_llm_evaluation(llm_evaluation: Dict[str, Any]) -> None:
    """
    Render LLM evaluation results.
    
    Args:
        llm_evaluation: Dictionary containing LLM evaluation results
    """
    if not llm_evaluation:
        st.warning("No LLM evaluation performed.")
        return
    
    # Overall feedback
    feedback = llm_evaluation.get("feedback", "No feedback provided.")
    st.write("**AI Feedback:**")
    st.write(feedback)
    
    # Detailed criteria if available
    criteria = llm_evaluation.get("criteria", {})
    if criteria:
        st.write("**Evaluation Criteria:**")
        
        for criterion, result in criteria.items():
            score = result.get("score", 0)
            comment = result.get("comment", "No comment")
            
            # Color code based on score
            if score >= 8:
                st.success(f"**{criterion}** ({score}/10): {comment}")
            elif score >= 6:
                st.warning(f"**{criterion}** ({score}/10): {comment}")
            else:
                st.error(f"**{criterion}** ({score}/10): {comment}")
    
    # Suggestions
    suggestions = llm_evaluation.get("suggestions", [])
    if suggestions:
        st.write("**Improvement Suggestions:**")
        for suggestion in suggestions:
            st.write(f"• {suggestion}")


def _render_raw_results(grading_results: Dict[str, Any]) -> None:
    """
    Render raw grading results as JSON.
    
    Args:
        grading_results: Complete grading results dictionary
    """
    st.write("**Complete Grading Results:**")
    st.json(grading_results)
    
    # Download button for results
    results_json = json.dumps(grading_results, indent=2)
    st.download_button(
        label="📥 Download Results JSON",
        data=results_json,
        file_name="grading_results.json",
        mime="application/json"
    )
